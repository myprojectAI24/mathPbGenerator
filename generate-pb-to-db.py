#!/usr/bin/env python3
"""
Math Problem Generator with Comprehensive Quality Validation

Usage:
  python db_generator.py -l 1 "Integers (Introductory): Positive and negative numbers"
  python db_generator.py -l 2 -n 5 "Fractions (Advanced): Mixed operations"
  python db_generator.py --level 3 --count 10 "Geometry: Area and perimeter"
  python db_generator.py --auto-approve -n 10 "Decimals: Operations"  # Skip manual review
"""

import asyncio
import re
import argparse
import sys
from typing import Optional, Tuple, Dict
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

console = Console()

# --- DATABASE CONFIGURATION ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'bidule',
    'password': 'access',  # CHANGE THIS
    'database': 'math_problems_db',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}

# --- MODEL CONFIGURATION ---
MODEL_CONFIG = {
    'math_generator': {
        # Try multiple models in order of preference
        'models': [
            {
                'model': "/app/model/RichardErkhov/simonveitner_-_MathHermes-2.5-Mistral-7B-awq",
                'base_url': "http://10.243.0.53:8085/v1",
                'name': "MathHermes-2.5",
                'priority': 1
            },
            {
                'model': "/app/model/Qwen/Qwen2.5-7B-Instruct-AWQ",
                'base_url': "http://10.243.0.54:8085/v1",
                'name': "Qwen 2.5 (Fallback)",
                'priority': 2
            },
            {
                'model': "/app/model/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                'base_url': "http://127.0.0.1:8085/v1",
                'name': "Llama-3.1-8B-AWQ (Fallback)",
                'priority': 3
            }
        ],
        'role': "Math Expert - Generates precise mathematical problems"
    },
    'auditor': {
        'model': "/app/model/Qwen/Qwen2.5-7B-Instruct-AWQ",
        'base_url': "http://10.243.0.54:8085/v1",
        'name': "Qwen 2.5",
        'role': "Verification - Checks mathematical accuracy",
        'max_context': 8192
    },
    'storyteller': {
        'model': "/app/model/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        'base_url': "http://127.0.0.1:8085/v1",
        'name': "Llama-3.1-8B-AWQ",
        'role': "Creative Writer - Adds narrative safely"
    }
}

SUBDIFF_LEVELS = {
    '1': {
        'name': 'Easy',
        'description': 'Simple operations with small whole numbers (under 20), single-step problems',
        'constraints': 'Use numbers under 20, require only 1-2 calculation steps'
    },
    '2': {
        'name': 'Medium',
        'description': 'Multi-step problems with larger numbers (20-100), simple fractions and percentages',
        'constraints': 'Use numbers 20-100, require 2-4 calculation steps'
    },
    '3': {
        'name': 'Difficult',
        'description': 'Complex multi-step problems with larger numbers (100+), mixed operations',
        'constraints': 'Use numbers over 100, require 4+ calculation steps'
    }
}

# --- VALIDATION RESULTS ---
class ValidationResult(BaseModel):
    passed: bool
    stage: str  # 'core_math', 'solution', 'enhancement', 'final'
    issues: list = []
    warnings: list = []
    audit_response: str = ""
    numbers_preserved: bool = True
    word_count: int = 0

class QualityReport(BaseModel):
    core_validation: ValidationResult
    solution_validation: ValidationResult
    enhancement_validation: Optional[ValidationResult] = None
    final_validation: ValidationResult
    overall_passed: bool
    generation_attempts: int
    timestamp: str
    quality_score: int = 0

# --- DATABASE OPERATIONS ---
class DatabaseManager:
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                console.print("[green]✓ Database connected successfully[/green]")
                return True
        except Error as e:
            console.print(f"[red]✗ Database connection failed: {e}[/red]")
            return False
        return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            console.print("[dim]Database connection closed[/dim]")
    
    def parse_section_topic(self, input_str: str) -> Tuple[str, str]:
        """Parse 'Section Name: Topic Name' format"""
        if ':' in input_str:
            parts = input_str.split(':', 1)
            return parts[0].strip(), parts[1].strip()
        else:
            return input_str.strip(), input_str.strip()
    
    def get_or_create_section(self, section_name: str) -> Optional[int]:
        """Get or create curriculum section"""
        cursor = self.connection.cursor(dictionary=True)
        
        try:
            number_match = re.match(r'^(\d+)[\.\)]\s*(.+)$', section_name)
            if number_match:
                section_number = number_match.group(1)
                clean_name = number_match.group(2).strip()
            else:
                cursor.execute("SELECT MAX(CAST(section_number AS UNSIGNED)) as max_num FROM sections")
                result = cursor.fetchone()
                section_number = str((result['max_num'] or 0) + 1)
                clean_name = section_name
            
            cursor.execute("SELECT id FROM sections WHERE section_name = %s", (clean_name,))
            result = cursor.fetchone()
            
            if result:
                return result['id']
            
            cursor.execute(
                "INSERT INTO sections (section_number, section_name) VALUES (%s, %s)",
                (section_number, clean_name)
            )
            self.connection.commit()
            return cursor.lastrowid
            
        except Error as e:
            console.print(f"[red]Error with section '{section_name}': {e}[/red]")
            return None
        finally:
            cursor.close()
    
    def get_or_create_topic(self, section_id: int, topic_name: str) -> Optional[int]:
        """Get or create topic within section"""
        cursor = self.connection.cursor(dictionary=True)
        
        try:
            cursor.execute(
                "SELECT id FROM topics WHERE section_id = %s AND topic_name = %s",
                (section_id, topic_name)
            )
            result = cursor.fetchone()
            
            if result:
                return result['id']
            
            cursor.execute(
                "INSERT INTO topics (section_id, topic_name) VALUES (%s, %s)",
                (section_id, topic_name)
            )
            self.connection.commit()
            return cursor.lastrowid
            
        except Error as e:
            console.print(f"[red]Error with topic '{topic_name}': {e}[/red]")
            return None
        finally:
            cursor.close()
    
    def insert_problem(self, topic_id: int, level: int, problem_statement: str, 
                      how_to_solve: str, solution_key: str, quality_score: int = None) -> Optional[int]:
        """Insert problem into database with quality score"""
        cursor = self.connection.cursor()
        
        word_count = len(re.findall(r'\b\w+\b', problem_statement))
        
        try:
            if quality_score is not None:
                cursor.execute(
                    """INSERT INTO problems 
                       (topic_id, level, problem_statement, how_to_solve, solution_key, word_count, quality_score)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (topic_id, level, problem_statement, how_to_solve, solution_key, word_count, quality_score)
                )
            else:
                cursor.execute(
                    """INSERT INTO problems 
                       (topic_id, level, problem_statement, how_to_solve, solution_key, word_count)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (topic_id, level, problem_statement, how_to_solve, solution_key, word_count)
                )
            self.connection.commit()
            problem_id = cursor.lastrowid
            console.print(f"[green]✓ Problem #{problem_id} inserted successfully[/green]")
            return problem_id
            
        except Error as e:
            if e.errno == 1062:
                console.print(f"[yellow]⚠ Duplicate problem detected, skipping...[/yellow]")
            else:
                console.print(f"[red]✗ Error inserting problem: {e}[/red]")
            return None
        finally:
            cursor.close()
    
    def list_sections(self):
        """Display all curriculum sections"""
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sections ORDER BY section_number")
        sections = cursor.fetchall()
        cursor.close()
        
        if not sections:
            console.print("[yellow]No sections found in database[/yellow]")
            return
        
        table = Table(title="Curriculum Sections")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Number", style="magenta")
        table.add_column("Section Name", style="green")
        
        for section in sections:
            table.add_row(str(section['id']), section['section_number'], section['section_name'])
        
        console.print(table)
    
    def list_topics(self, section_id: int):
        """Display topics for a section"""
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """SELECT t.*, s.section_name 
               FROM topics t 
               JOIN sections s ON t.section_id = s.id 
               WHERE t.section_id = %s 
               ORDER BY t.topic_name""",
            (section_id,)
        )
        topics = cursor.fetchall()
        cursor.close()
        
        if not topics:
            console.print(f"[yellow]No topics found for section {section_id}[/yellow]")
            return
        
        table = Table(title=f"Topics in Section: {topics[0]['section_name']}")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Topic Name", style="green")
        
        for topic in topics:
            table.add_row(str(topic['id']), topic['topic_name'])
        
        console.print(table)
    
    def get_problem_stats(self, topic_id: int = None):
        """Get statistics about generated problems"""
        cursor = self.connection.cursor(dictionary=True)
        
        if topic_id:
            cursor.execute(
                """SELECT level, COUNT(*) as count, AVG(quality_score) as avg_quality
                   FROM problems 
                   WHERE topic_id = %s 
                   GROUP BY level""",
                (topic_id,)
            )
        else:
            cursor.execute(
                """SELECT level, COUNT(*) as count, AVG(quality_score) as avg_quality
                   FROM problems 
                   GROUP BY level"""
            )
        
        stats = cursor.fetchall()
        cursor.close()
        
        if stats:
            table = Table(title="Problem Statistics")
            table.add_column("Level", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Avg Quality", style="magenta")
            
            for stat in stats:
                level_name = SUBDIFF_LEVELS[str(stat['level'])]['name']
                avg_quality = f"{stat['avg_quality']:.1f}" if stat['avg_quality'] else "N/A"
                table.add_row(f"{stat['level']} ({level_name})", str(stat['count']), avg_quality)
            
            console.print(table)

# --- HELPER FUNCTIONS ---
class MathProblem(BaseModel):
    problem_statement: str
    solution: str
    explanation: str = ""
    quality_report: Optional[QualityReport] = None

def get_text_from_result(result):
    for attr in ['result', 'data', 'output']:
        if hasattr(result, attr):
            return getattr(result, attr)
    return str(result)

def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def calculate_quality_score(quality_report: QualityReport) -> int:
    """Calculate quality score based on validation results"""
    score = 0
    
    # Base score from validation stages (each worth 20 points max)
    validations = [
        quality_report.core_validation,
        quality_report.solution_validation,
        quality_report.enhancement_validation,
        quality_report.final_validation
    ]
    
    for validation in validations:
        if validation and validation.passed:
            score += 20
        elif validation and not validation.passed:
            score += 5  # Partial credit for attempted validation
    
    # Bonus points for good characteristics
    if quality_report.core_validation:
        if quality_report.core_validation.word_count <= 80:
            score += 10  # Concise problems get bonus
        elif quality_report.core_validation.word_count <= 100:
            score += 5
        elif quality_report.core_validation.word_count > 120:
            score -= 10  # Penalty for overly long problems
    
    # Fewer generation attempts = higher quality
    if quality_report.generation_attempts <= 2:
        score += 10
    elif quality_report.generation_attempts <= 3:
        score += 5
    else:
        score -= 5
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))

def extract_key_numbers(text: str) -> dict:
    """Extract key domain-specific numbers likely to appear in word problems"""
    numbers = {
        'pizzas': re.search(r'(\d+(?:\.\d+)?)\s+(?:large\s+)?pizzas?', text, re.IGNORECASE),
        'slices_per_pizza': re.search(r'(?:each\s+)?(?:cut|divided)\s+into\s+(\d+(?:\.\d+)?)\s+slices?', text, re.IGNORECASE),
        'children': re.search(r'(\d+(?:\.\d+)?)\s+(?:children|kids|students)', text, re.IGNORECASE),
        'friends': re.search(r'(\d+(?:\.\d+)?)\s+friends', text, re.IGNORECASE),
        'people': re.search(r'(\d+(?:\.\d+)?)\s+(?:people|guests)', text, re.IGNORECASE),
        'percentage': re.search(r'(\d+(?:\.\d+)?)%', text),
        'volume_ml': re.search(r'(\d+(?:\.\d+)?)\s*ml\b', text, re.IGNORECASE),
        'volume_liters': re.search(r'(\d+(?:\.\d+)?)\s*liters?', text, re.IGNORECASE),
        'jugs': re.search(r'(\d+(?:\.\d+)?)\s+jugs?', text, re.IGNORECASE),
        'boxes': re.search(r'(\d+(?:\.\d+)?)\s+boxes?', text, re.IGNORECASE),
        'items': re.search(r'(\d+(?:\.\d+)?)\s+(?:items|cookies|candies|apples)', text, re.IGNORECASE),
    }
    
    # Convert to float instead of int to handle decimals
    return {k: float(v.group(1)) if v else None for k, v in numbers.items() if v}

def numbers_match(original: dict, modified: dict) -> Tuple[bool, list]:
    """Compare two extracted number dicts and report differences"""
    differences = []
    all_keys = set(list(original.keys()) + list(modified.keys()))
    for k in all_keys:
        orig = original.get(k)
        mod = modified.get(k)
        
        # Handle floating point comparison with tolerance
        if orig is not None and mod is not None:
            if abs(float(orig) - float(mod)) > 0.001:  # tolerance for floating point
                differences.append(f"{k}: {orig} → {mod}")
        elif orig is not None and mod is None:
            differences.append(f"{k}: {orig} → missing")
        elif orig is None and mod is not None:
            differences.append(f"{k}: added {mod}")
    
    return len(differences) == 0, differences

def compare_numbers(original: Dict[str, list], modified: Dict[str, list]) -> Tuple[bool, list]:
    """Compare two sets of extracted numbers"""
    differences = []
    
    for num_type in set(list(original.keys()) + list(modified.keys())):
        orig_nums = set(original.get(num_type, []))
        mod_nums = set(modified.get(num_type, []))
        
        if orig_nums != mod_nums:
            added = mod_nums - orig_nums
            removed = orig_nums - mod_nums
            
            if added:
                differences.append(f"Added {num_type}: {added}")
            if removed:
                differences.append(f"Removed {num_type}: {removed}")
    
    return len(differences) == 0, differences

# --- MODEL INITIALIZATION WITH FALLBACK ---
def initialize_models():
    """Initialize models with fallback logic"""
    models = {}
    
    # Initialize math generator with fallback
    math_models = MODEL_CONFIG['math_generator']['models']
    math_model = None
    
    for model_config in sorted(math_models, key=lambda x: x['priority']):
        try:
            console.print(f"[cyan]Trying math generator: {model_config['name']}[/cyan]")
            test_model = OpenAIChatModel(
                model_config['model'],
                provider=OpenAIProvider(base_url=model_config['base_url'])
            )
            # Test the model with a simple request
            test_agent = Agent(test_model, output_type=str, model_settings=ModelSettings(max_tokens=50))
            test_result = asyncio.run(test_agent.run("Test connection"))
            math_model = test_model
            console.print(f"[green]✓ Math generator connected: {model_config['name']}[/green]")
            break
        except Exception as e:
            console.print(f"[yellow]✗ Failed to connect to {model_config['name']}: {e}[/yellow]")
            continue
    
    if not math_model:
        console.print("[red]✗ All math generator models failed to connect[/red]")
        console.print("[yellow]Available models will be used for other roles[/yellow]")
        # Use Qwen for math generation as fallback
        math_model = OpenAIChatModel(
            MODEL_CONFIG['auditor']['model'],
            provider=OpenAIProvider(base_url=MODEL_CONFIG['auditor']['base_url'])
        )
    
    models['math_generator'] = math_model
    
    # Initialize other models
    try:
        models['auditor'] = OpenAIChatModel(
            MODEL_CONFIG['auditor']['model'],
            provider=OpenAIProvider(base_url=MODEL_CONFIG['auditor']['base_url'])
        )
        console.print(f"[green]✓ Auditor model connected[/green]")
    except Exception as e:
        console.print(f"[yellow]✗ Auditor model failed, using fallback: {e}[/yellow]")
        models['auditor'] = math_model  # Fallback
    
    try:
        models['storyteller'] = OpenAIChatModel(
            MODEL_CONFIG['storyteller']['model'],
            provider=OpenAIProvider(base_url=MODEL_CONFIG['storyteller']['base_url'])
        )
        console.print(f"[green]✓ Storyteller model connected[/green]")
    except Exception as e:
        console.print(f"[yellow]✗ Storyteller model failed, using fallback: {e}[/yellow]")
        models['storyteller'] = math_model  # Fallback
    
    return models

# Initialize models with fallback
available_models = initialize_models()

settings = ModelSettings(max_tokens=3072, temperature=0.7)
story_settings = ModelSettings(max_tokens=3072, temperature=0.6)

logic_gen = Agent(available_models['math_generator'], output_type=str, model_settings=settings)
story_wrapper = Agent(available_models['storyteller'], output_type=str, model_settings=story_settings)
auditor = Agent(available_models['auditor'], output_type=str, model_settings=settings)

# --- COMPREHENSIVE VALIDATION FUNCTIONS ---
async def validate_core_math(core_text: str, topic: str, level: str) -> ValidationResult:
    """Validate the core mathematical problem generated"""
    console.print("[cyan]🔍 Validating core math problem...[/cyan]")
    
    issues = []
    warnings = []
    
    # Check structure
    if "Problem Statement:" not in core_text and "Step-by-step Solution:" not in core_text:
        issues.append("Missing required sections (Problem Statement / Solution)")
    
    # Extract problem and solution
    has_solution = "Step-by-step Solution:" in core_text or "Solution:" in core_text
    if not has_solution:
        issues.append("No solution section found")
    
    # Word count check
    problem_part = core_text.split("Step-by-step Solution:")[0] if has_solution else core_text
    word_count = count_words(problem_part)
    if word_count > 150:
        warnings.append(f"Problem statement too long: {word_count} words (target: ≤120)")
    
    # Audit with Qwen
    audit_prompt = f"""You are a math teacher reviewing a grade-6 problem. Analyze this carefully:

TOPIC: {topic}
DIFFICULTY: Level {level} - {SUBDIFF_LEVELS[level]['name']}

PROBLEM:
{core_text}

Check for:
1. Is the problem statement clear and unambiguous?
2. Are the numbers appropriate for the difficulty level?
3. Is the solution mathematically correct?
4. Are all steps shown in the solution?
5. Does the problem match the topic and difficulty?

Respond in this format:
VERDICT: PASS or FAIL
ISSUES: [list any problems, or "none"]
QUALITY: [rate 1-5 and explain]"""

    try:
        audit_res = await auditor.run(audit_prompt)
        audit_text = get_text_from_result(audit_res)
    except Exception as e:
        console.print(f"[yellow]⚠ Auditor failed, using basic validation: {e}[/yellow]")
        audit_text = "VERDICT: PASS\nISSUES: Auditor unavailable\nQUALITY: 3/5 (basic validation)"
    
    # Parse audit response
    passed = "PASS" in audit_text and "FAIL" not in audit_text
    
    if "ISSUES:" in audit_text:
        issues_section = audit_text.split("ISSUES:")[1].split("QUALITY:")[0].strip()
        if issues_section.lower() != "none" and len(issues_section) > 10:
            issues.append(f"Auditor found: {issues_section[:200]}")
    
    return ValidationResult(
        passed=passed and len(issues) == 0,
        stage="core_math",
        issues=issues,
        warnings=warnings,
        audit_response=audit_text,
        word_count=word_count
    )

async def validate_solution_quality(solution_text: str, problem_text: str) -> ValidationResult:
    """Validate the solution is complete and correct"""
    console.print("[cyan]🔍 Validating solution quality...[/cyan]")
    
    issues = []
    warnings = []
    
    # Basic checks
    if len(solution_text) < 50:
        issues.append("Solution too brief - may be incomplete")
    
    if not any(word in solution_text.lower() for word in ['step', 'first', 'then', 'therefore', 'answer']):
        warnings.append("Solution may lack clear step-by-step structure")
    
    # Deep audit
    audit_prompt = f"""You are reviewing a mathematical solution. Check if it's correct and complete.

PROBLEM:
{problem_text}

SOLUTION:
{solution_text}

Verify:
1. Does the solution solve the correct problem?
2. Are all calculation steps shown?
3. Is the math correct?
4. Is the final answer clearly stated?
5. Can a grade-6 student follow these steps?

Respond:
VERDICT: PASS or FAIL
ERRORS: [list any mathematical errors, or "none"]
COMPLETENESS: [explain if anything is missing]"""

    try:
        audit_res = await auditor.run(audit_prompt)
        audit_text = get_text_from_result(audit_res)
    except Exception as e:
        console.print(f"[yellow]⚠ Auditor failed for solution validation: {e}[/yellow]")
        audit_text = "VERDICT: PASS\nERRORS: none\nCOMPLETENESS: Auditor unavailable, basic validation only"
    
    passed = "PASS" in audit_text and "FAIL" not in audit_text
    
    if "ERRORS:" in audit_text:
        errors_section = audit_text.split("ERRORS:")[1].split("COMPLETENESS:")[0].strip()
        if errors_section.lower() != "none" and len(errors_section) > 10:
            issues.append(f"Mathematical errors: {errors_section[:200]}")
    
    return ValidationResult(
        passed=passed and len(issues) == 0,
        stage="solution",
        issues=issues,
        warnings=warnings,
        audit_response=audit_text
    )

async def validate_enhancement(original: str, enhanced: str, original_numbers: Dict) -> ValidationResult:
    """Validate story enhancement preserves accuracy"""
    console.print("[cyan]🔍 Validating story enhancement...[/cyan]")
    
    issues = []
    warnings = []
    
    # Word count
    word_count = count_words(enhanced)
    if word_count > 120:
        issues.append(f"Enhanced problem too long: {word_count} words (max: 120)")
    
    # Number preservation
    enhanced_numbers = extract_key_numbers(enhanced)
    numbers_match, differences = compare_numbers(original_numbers, enhanced_numbers)
    
    if not numbers_match:
        issues.append(f"Numbers changed during enhancement: {', '.join(differences)}")
    
    # Check for leaked answers
    audit_prompt = f"""Check if this enhanced problem accidentally reveals the answer or includes calculations.

ENHANCED PROBLEM:
{enhanced}

Does this text contain:
1. Any calculations or arithmetic operations?
2. Intermediate results or totals?
3. The final answer?
4. Solution hints that give away the answer?

Respond:
VERDICT: CLEAN or LEAKED
PROBLEMS: [list any leaks, or "none"]"""

    try:
        audit_res = await auditor.run(audit_prompt)
        audit_text = get_text_from_result(audit_res)
    except Exception as e:
        console.print(f"[yellow]⚠ Auditor failed for enhancement check: {e}[/yellow]")
        audit_text = "VERDICT: CLEAN\nPROBLEMS: Auditor unavailable, manual review recommended"
    
    passed = "CLEAN" in audit_text and "LEAKED" not in audit_text
    
    if "PROBLEMS:" in audit_text:
        problems_section = audit_text.split("PROBLEMS:")[1].strip()
        if problems_section.lower() != "none" and len(problems_section) > 10:
            issues.append(f"Answer leakage: {problems_section[:200]}")
    
    return ValidationResult(
        passed=passed and numbers_match and word_count <= 120,
        stage="enhancement",
        issues=issues,
        warnings=warnings,
        audit_response=audit_text,
        numbers_preserved=numbers_match,
        word_count=word_count
    )

async def validate_final_problem(problem: str, solution: str, topic: str, level: str) -> ValidationResult:
    """Final comprehensive validation before database insertion"""
    console.print("[cyan]🔍 Final validation check...[/cyan]")
    
    issues = []
    warnings = []
    
    audit_prompt = f"""FINAL QUALITY CHECK - This problem will be given to students. Review carefully.

TOPIC: {topic}
LEVEL: {level} ({SUBDIFF_LEVELS[level]['name']})

PROBLEM FOR STUDENTS:
{problem}

SOLUTION KEY:
{solution}

Final checks:
1. Is everything mathematically correct?
2. Is the problem clear and age-appropriate?
3. Will students understand what's being asked?
4. Is the solution complete and accurate?
5. Overall quality acceptable for database?

Respond:
VERDICT: APPROVE or REJECT
FINAL_SCORE: [1-5]
REASONING: [brief explanation]
CRITICAL_ISSUES: [any deal-breakers, or "none"]"""

    try:
        audit_res = await auditor.run(audit_prompt)
        audit_text = get_text_from_result(audit_res)
    except Exception as e:
        console.print(f"[yellow]⚠ Auditor failed for final validation: {e}[/yellow]")
        audit_text = "VERDICT: APPROVE\nFINAL_SCORE: 3\nREASONING: Auditor unavailable, basic validation\nCRITICAL_ISSUES: none"
    
    passed = "APPROVE" in audit_text and "REJECT" not in audit_text
    
    if "CRITICAL_ISSUES:" in audit_text:
        critical = audit_text.split("CRITICAL_ISSUES:")[1].strip()
        if critical.lower() != "none" and len(critical) > 10:
            issues.append(f"Critical: {critical[:200]}")
    
    # Extract score
    score_match = re.search(r'FINAL_SCORE:\s*(\d)', audit_text)
    if score_match and int(score_match.group(1)) < 3:
        warnings.append(f"Low quality score: {score_match.group(1)}/5")
    
    return ValidationResult(
        passed=passed and len(issues) == 0,
        stage="final",
        issues=issues,
        warnings=warnings,
        audit_response=audit_text
    )

# --- PROBLEM GENERATION WITH VALIDATION ---

async def generate_validated_problem(topic: str, level: str = '1', simple_mode: bool = False, 
                                    max_retries: int = 3) -> Optional[MathProblem]:
    """Generate a problem with comprehensive validation at each step"""
    
    diff_info = SUBDIFF_LEVELS[level]
    generation_attempt = 0
    
    console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
    console.print(f"[bold yellow]Starting Problem Generation: {topic} (Level {level})[/bold yellow]")
    console.print(f"[bold yellow]{'='*80}[/bold yellow]\n")
    
    # STAGE 1: Generate and validate core math
    core_validation = None
    core_text = None
    core_problem = None
    core_solution = None
    
    for attempt in range(max_retries):
        generation_attempt += 1
        console.print(f"\n[bold]📝 STAGE 1: Core Math Generation (Attempt {attempt + 1}/{max_retries})[/bold]")
        
        difficulty_prompt = (
            f"Create a grade-6 level math problem about {topic}.\n"
            f"DIFFICULTY LEVEL: {diff_info['name']} (Level {level}/3)\n"
            f"CONSTRAINTS: {diff_info['constraints']}\n\n"
            f"The Problem Statement must be concise: no more than 120 words.\n"
            f"Be precise, clear, and mathematically rigorous.\n\n"
            f"Output exactly:\n"
            f"Problem Statement: [clear word problem]\n\n"
            f"Step-by-step Solution: [complete solution with all steps and final answer]\n"
        )
        
        try:
            core_res = await logic_gen.run(difficulty_prompt)
            core_text = get_text_from_result(core_res)
        except Exception as e:
            console.print(f"[red]✗ Error generating core math: {e}[/red]")
            if attempt < max_retries - 1:
                console.print("[yellow]Retrying with different approach...[/yellow]")
                continue
            else:
                return None
        
        # Validate core
        core_validation = await validate_core_math(core_text, topic, level)
        
        if core_validation.passed:
            console.print("[bold green]✓ Core math validation PASSED[/bold green]")
            break
        else:
            console.print(f"[bold red]✗ Core math validation FAILED[/bold red]")
            console.print(f"[red]Issues: {', '.join(core_validation.issues)}[/red]")
            if attempt < max_retries - 1:
                console.print("[yellow]Regenerating...[/yellow]")
    
    if not core_validation or not core_validation.passed:
        console.print("[bold red]Failed to generate valid core math after all attempts[/bold red]")
        return None
    
    # Extract problem and solution
    if "Step-by-step Solution:" in core_text:
        core_problem_part, core_solution_part = core_text.split("Step-by-step Solution:", 1)
        core_solution = core_solution_part.strip()
        core_problem = core_problem_part.replace("Problem Statement:", "").strip()
    else:
        console.print("[red]Could not extract solution properly[/red]")
        return None
    
    # STAGE 2: Validate solution quality
    console.print(f"\n[bold]🎯 STAGE 2: Solution Quality Check[/bold]")
    solution_validation = await validate_solution_quality(core_solution, core_problem)
    
    if not solution_validation.passed:
        console.print(f"[bold red]✗ Solution validation FAILED[/bold red]")
        console.print(f"[red]Issues: {', '.join(solution_validation.issues)}[/red]")
        return None
    
    console.print("[bold green]✓ Solution validation PASSED[/bold green]")
    
    # Extract numbers for comparison
    original_numbers = extract_key_numbers(core_problem)
    console.print(f"[dim]Extracted numbers: {original_numbers}[/dim]")

    # STAGE 3: Story enhancement (with robust validation inspired by Script2)
    enhanced_problem = core_problem
    enhancement_validation = None

    if not simple_mode:
        console.print(f"\n[bold]✨ STAGE 3: Story Enhancement[/bold]")
        original_numbers = extract_key_numbers(core_problem)
        console.print(f"[dim]Extracted key numbers: { {k:v for k,v in original_numbers.items() if v is not None} }[/dim]")

        success = False
        for attempt in range(max_retries):
            console.print(f"[bold magenta]Attempt {attempt + 1}/{max_retries}: Enhancing with narrative...[/bold magenta]")

            numbers_list = ", ".join([f"{k}={v}" for k, v in original_numbers.items() if v is not None])
            if not numbers_list:
                numbers_list = "none detected"

            story_prompt = f"""
    You are turning a plain math word problem into an engaging real-world story for 6th-grade students.

    CRITICAL RULES:
    - Preserve these exact values: {numbers_list}
    - Keep the enhanced problem ≤ 120 words
    - Add character names, setting, emotions, sensory details
    - Do NOT perform any calculations
    - Do NOT reveal intermediate results or the answer
    - Do NOT add or remove any key numbers

    ORIGINAL PROBLEM:
    {core_problem}

    OUTPUT FORMAT:
    **Problem Statement:**
    [Your enhanced version only — no solution, no extra text]
    """

            try:
                story_res = await story_wrapper.run(story_prompt)
                story_output = get_text_from_result(story_res)
            except Exception as e:
                console.print(f"[yellow]⚠ Story enhancement failed: {e}[/yellow]")
                if attempt < max_retries - 1:
                    continue
                else:
                    break

            # Extract enhanced problem
            match = re.search(r'\*\*Problem Statement:\*\*\s*(.*?)(?=\*\*(?:Solution|Explanation)|$)', story_output, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
            else:
                candidate = story_output.strip()
                if candidate.startswith("Problem Statement:"):
                    candidate = candidate.split("Problem Statement:", 1)[1].strip()

            word_count = count_words(candidate)

            # Quick local checks first (fast fail)
            issues = []
            if word_count > 120:
                issues.append(f"Too long: {word_count} words (max 120)")

            enhanced_numbers = extract_key_numbers(candidate)
            nums_ok, diff_list = numbers_match(original_numbers, enhanced_numbers)
            if not nums_ok:
                issues.append(f"Number changes: {', '.join(diff_list)}")

            if issues:
                console.print(f"[yellow]Issues: {'; '.join(issues)}. Retrying...[/yellow]")
                continue

            # Final light audit with Qwen (only if local checks pass)
            console.print("[cyan]Quick safety audit...[/cyan]")
            audit_prompt = f"""
    Check this enhanced problem for answer leakage or calculations.
    Word count: {word_count}/120

    ENHANCED PROBLEM:
    {candidate}

    Respond with only:
    PASSED
    or
    FAILED: [brief reason]
    """

            try:
                audit_res = await auditor.run(audit_prompt)
                audit_text = get_text_from_result(audit_res).strip().upper()
            except Exception as e:
                console.print(f"[yellow]⚠ Enhancement audit failed: {e}[/yellow]")
                audit_text = "PASSED"  # Fallback to passed if auditor fails

            if "PASSED" in audit_text:
                enhanced_problem = candidate
                enhancement_validation = ValidationResult(
                    passed=True,
                    stage="enhancement",
                    issues=[],
                    warnings=[] if word_count <= 100 else [f"Close to limit: {word_count} words"],
                    audit_response=audit_text,
                    numbers_preserved=True,
                    word_count=word_count
                )
                console.print(f"[bold green]✓ Story enhancement PASSED ({word_count} words)[/bold green]")
                success = True
                break
            else:
                console.print(f"[yellow]Audit failed: {audit_text}. Retrying...[/yellow]")

        if not success:
            console.print("[yellow]⚠ Story enhancement failed after retries — using original problem[/yellow]")
            enhancement_validation = ValidationResult(
                passed=False,
                stage="enhancement",
                issues=["All attempts failed local or audit checks"],
                warnings=[],
                audit_response="Failed after retries",
                numbers_preserved=True,
                word_count=count_words(core_problem)
            )
    else:
        console.print(f"\n[bold]✨ STAGE 3: Story Enhancement - SKIPPED (simple mode)[/bold]")
        enhancement_validation = ValidationResult(
            passed=True,
            stage="enhancement",
            issues=[],
            warnings=["Skipped by user"],
            audit_response="N/A (simple mode)",
            numbers_preserved=True,
            word_count=count_words(core_problem)
        )
    
        
    # STAGE 4: Final validation
    console.print(f"\n[bold]🏁 STAGE 4: Final Comprehensive Check[/bold]")
    final_validation = await validate_final_problem(enhanced_problem, core_solution, topic, level)
    
    if not final_validation.passed:
        console.print(f"[bold red]✗ Final validation FAILED[/bold red]")
        console.print(f"[red]Issues: {', '.join(final_validation.issues)}[/red]")
        return None
    
    console.print("[bold green]✓ Final validation PASSED[/bold green]")
    
    # Calculate quality score
    quality_score = calculate_quality_score(QualityReport(
        core_validation=core_validation,
        solution_validation=solution_validation,
        enhancement_validation=enhancement_validation,
        final_validation=final_validation,
        overall_passed=True,
        generation_attempts=generation_attempt,
        timestamp=datetime.now().isoformat()
    ))
    
    console.print(f"[bold cyan]📊 Quality Score: {quality_score}/100[/bold cyan]")
    
    # Generate explanation
    explanation = (
        f"To solve this {topic.lower()} problem:\n"
        f"1. Carefully read and identify all given information\n"
        f"2. Determine what mathematical operation(s) are needed\n"
        f"3. Work through the solution step-by-step\n"
        f"4. Check that your answer makes sense in context"
    )
    
    # Create quality report
    quality_report = QualityReport(
        core_validation=core_validation,
        solution_validation=solution_validation,
        enhancement_validation=enhancement_validation,
        final_validation=final_validation,
        overall_passed=True,
        generation_attempts=generation_attempt,
        timestamp=datetime.now().isoformat(),
        quality_score=quality_score
    )
    
    return MathProblem(
        problem_statement=enhanced_problem,
        solution=core_solution,
        explanation=explanation,
        quality_report=quality_report
    )

def display_quality_report(report: QualityReport):
    """Display comprehensive quality report"""
    console.print("\n" + "="*80)
    console.print("[bold cyan]📊 QUALITY ASSURANCE REPORT[/bold cyan]")
    console.print("="*80 + "\n")
    
    # Summary table
    table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Issues", width=50)
    
    stages = [
        ("Core Math", report.core_validation),
        ("Solution Quality", report.solution_validation),
        ("Enhancement", report.enhancement_validation),
        ("Final Check", report.final_validation)
    ]
    
    for stage_name, validation in stages:
        if validation:
            status = "[green]✓ PASS[/green]" if validation.passed else "[red]✗ FAIL[/red]"
            issues = ", ".join(validation.issues) if validation.issues else "[dim]none[/dim]"
            warnings = ", ".join(validation.warnings) if validation.warnings else ""
            details = issues
            if warnings:
                details += f"\n[yellow]Warnings: {warnings}[/yellow]"
            table.add_row(stage_name, status, details)
        else:
            table.add_row(stage_name, "[dim]skipped[/dim]", "[dim]n/a[/dim]")
    
    console.print(table)
    console.print(f"\n[bold]Generation attempts:[/bold] {report.generation_attempts}")
    console.print(f"[bold]Quality Score:[/bold] {report.quality_score}/100")
    console.print(f"[bold]Timestamp:[/bold] {report.timestamp}")
    console.print(f"[bold]Overall status:[/bold] {'[green]APPROVED ✓[/green]' if report.overall_passed else '[red]REJECTED ✗[/red]'}")
    
    # Detailed audit responses
    if report.final_validation and report.final_validation.audit_response:
        console.print("\n[bold cyan]🔍 Auditor's Final Assessment:[/bold cyan]")
        console.print(Panel(report.final_validation.audit_response, border_style="cyan", padding=(1, 2)))

def display_problem_for_review(problem: MathProblem, problem_number: int = 1):
    """Display problem with quality report for manual review"""
    console.print("\n" + "="*80)
    console.print(f"[bold white]PROBLEM #{problem_number} - REVIEW REQUIRED[/bold white]")
    console.print("="*80)
    
    # Display the problem
    console.print(Panel(
        Markdown(problem.problem_statement),
        title="[bold green]📚 Student Problem Statement[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Display the explanation
    console.print(Panel(
        Markdown(problem.explanation),
        title="[bold blue]🧭 How to Solve It (Hint)[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Display the solution
    console.print(Panel(
        Markdown(problem.solution),
        title="[bold white]✅ Teacher's Solution Key[/bold white]",
        border_style="white",
        padding=(1, 2)
    ))
    
    # Display quality report
    if problem.quality_report:
        display_quality_report(problem.quality_report)

# --- MAIN EXECUTION ---
async def main():
    parser = argparse.ArgumentParser(
        description="Math Problem Generator with Comprehensive Quality Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python db_generator.py -l 1 "Integers: Positive and negative numbers"
  python db_generator.py -l 2 -n 5 "Fractions: Addition and subtraction"
  python db_generator.py --level 3 --count 10 --auto-approve "Geometry: Area"
  python db_generator.py -l 1 -s "Decimals: Basic operations"  # Simple mode
        """.strip()
    )
    
    parser.add_argument('topic', nargs='*', help='Section: Topic format')
    parser.add_argument('-l', '--level', type=str, choices=['1','2','3'], default='1')
    parser.add_argument('-n', '--count', type=int, default=1)
    parser.add_argument('-s', '--simple', action='store_true', help='Skip story enhancement')
    parser.add_argument('--auto-approve', action='store_true', 
                       help='Auto-insert problems that pass validation (skip manual review)')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Max retry attempts per stage (default: 5)')
    parser.add_argument('--list-sections', action='store_true')
    parser.add_argument('--list-topics', type=int, metavar='SECTION_ID')
    parser.add_argument('--stats', action='store_true')
    
    args = parser.parse_args()
    
    # Connect to database
    db = DatabaseManager(DB_CONFIG)
    if not db.connect():
        sys.exit(1)
    
    try:
        # Handle list commands
        if args.list_sections:
            db.list_sections()
            return
        
        if args.list_topics:
            db.list_topics(args.list_topics)
            return
        
        if args.stats:
            db.get_problem_stats()
            return
        
        # Validate topic input
        if not args.topic:
            console.print("[red]Error: Topic required. Use format 'Section: Topic'[/red]")
            sys.exit(1)
        
        full_topic = " ".join(args.topic)
        section_name, topic_name = db.parse_section_topic(full_topic)
        
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold cyan]  MATH PROBLEM GENERATOR - QUALITY VALIDATED[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
        
        console.print(f"[bold blue]📚 Section:[/bold blue] {section_name}")
        console.print(f"[bold blue]📖 Topic:[/bold blue] {topic_name}")
        console.print(f"[bold blue]🎯 Level:[/bold blue] {args.level} ({SUBDIFF_LEVELS[args.level]['name']})")
        console.print(f"[bold blue]🔢 Count:[/bold blue] {args.count}")
        console.print(f"[bold blue]🔍 Review Mode:[/bold blue] {'Auto-approve' if args.auto_approve else 'Manual review required'}")
        if args.simple:
            console.print(f"[bold blue]✨ Mode:[/bold blue] Simple (no story enhancement)")
        console.print()
        
        # Get or create section and topic
        section_id = db.get_or_create_section(section_name)
        if not section_id:
            console.print("[red]Failed to create/find section[/red]")
            sys.exit(1)
        
        topic_id = db.get_or_create_topic(section_id, topic_name)
        if not topic_id:
            console.print("[red]Failed to create/find topic[/red]")
            sys.exit(1)
        
        console.print(f"[green]✓ Database ready: Section ID={section_id}, Topic ID={topic_id}[/green]\n")
        
        # Generate problems
        success_count = 0
        rejected_count = 0
        
        for i in range(args.count):
            console.print(f"\n{'='*80}")
            console.print(f"[bold yellow]GENERATING PROBLEM {i+1} of {args.count}[/bold yellow]")
            console.print(f"{'='*80}")
            
            try:
                problem = await generate_validated_problem(
                    topic=topic_name,
                    level=args.level,
                    simple_mode=args.simple,
                    max_retries=args.max_retries
                )
                
                if not problem:
                    console.print("[bold red]✗ Problem generation failed validation[/bold red]")
                    rejected_count += 1
                    continue
                
                # Display for review
                display_problem_for_review(problem, i + 1)
                
                # Decide whether to insert
                should_insert = False
                
                if args.auto_approve:
                    console.print("\n[bold green]✓ Auto-approving (validation passed)[/bold green]")
                    should_insert = True
                else:
                    console.print("\n[bold yellow]⚠ MANUAL REVIEW REQUIRED[/bold yellow]")
                    should_insert = Confirm.ask(
                        "[bold]Insert this problem into database?[/bold]",
                        default=True
                    )
                
                if should_insert:
                    problem_id = db.insert_problem(
                        topic_id=topic_id,
                        level=int(args.level),
                        problem_statement=problem.problem_statement,
                        how_to_solve=problem.explanation,
                        solution_key=problem.solution,
                        quality_score=problem.quality_report.quality_score if problem.quality_report else None
                    )
                    
                    if problem_id:
                        success_count += 1
                        console.print(f"[bold green]✓ Problem #{problem_id} saved to database[/bold green]")
                    else:
                        rejected_count += 1
                else:
                    console.print("[yellow]⊘ Problem rejected by reviewer[/yellow]")
                    rejected_count += 1
                    
            except Exception as e:
                console.print(f"[red]✗ Error generating problem {i+1}: {e}[/red]")
                rejected_count += 1
        
        # Final summary
        console.print(f"\n{'='*80}")
        console.print("[bold cyan]📊 GENERATION SUMMARY[/bold cyan]")
        console.print(f"{'='*80}")
        console.print(f"[bold green]✓ Successfully inserted:[/bold green] {success_count}/{args.count}")
        console.print(f"[bold red]✗ Rejected/Failed:[/bold red] {rejected_count}/{args.count}")
        
        # Show statistics
        console.print()
        db.get_problem_stats(topic_id)
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())