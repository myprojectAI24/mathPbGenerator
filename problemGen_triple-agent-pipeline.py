# # Easy level (default)
# python script.py

# # Medium level
# python script.py --level 2
# # or short form
# python script.py -l 2

# # Difficult level with custom topic
# python script.py --level 3 "dividing treasure among pirates using fractions and percentages"

# # Verbose + simple mode + level 2
# python script.py -l 2 -v -s

# # Custom topic only (keeps default level 1)
# python script.py "a school bake sale with costs and profits"

import asyncio
import re
import argparse
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# --- CONFIGURATION ---
MODEL_CONFIG = {
    'math_generator': {
        'model': "/app/model/RichardErkhov/simonveitner_-_MathHermes-2.5-Mistral-7B-awq",
        'base_url': "http://10.243.0.53:8085/v1",
        'name': "MathHermes-2.5",
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

# Sub-difficulty level definitions
SUBDIFF_LEVELS = {
    '1': {
        'name': 'Easy',
        'description': 'Simple operations with small whole numbers (under 20), single-step problems, basic fractions (halves, quarters)',
        'constraints': 'Use numbers under 20, require only 1-2 calculation steps, avoid complex fractions or percentages'
    },
    '2': {
        'name': 'Medium',
        'description': 'Multi-step problems with larger numbers (20-100), simple fractions and percentages, basic word problem complexity',
        'constraints': 'Use numbers 20-100, require 2-4 calculation steps, may include simple fractions like 1/3, 1/4, 1/8 and percentages like 25%, 50%'
    },
    '3': {
        'name': 'Difficult',
        'description': 'Complex multi-step problems with larger numbers (100+), mixed operations, various fractions and percentages, multiple concepts',
        'constraints': 'Use numbers over 100, require 4+ calculation steps, include mixed fractions, various percentages, combine multiple mathematical concepts'
    }
}

# --- 1. DATA SCHEMA ---
class MathProblem(BaseModel):
    problem_statement: str
    solution: str
    explanation: str = ""

# --- 2. HELPERS ---
def get_text_from_result(result):
    for attr in ['result', 'data', 'output']:
        if hasattr(result, attr):
            return getattr(result, attr)
    return str(result)

def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def parse_text_problem(text: str) -> MathProblem:
    text = re.sub(r'^(FIXED VERSION:|PASSED:)\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = text.strip()

    problem_match = re.search(r'\*\*Problem Statement:\*\*\s*(.*?)(?=\*\*(?:Solution|How to Solve It|Explanation):\*\*|$)', text, re.IGNORECASE | re.DOTALL)
    problem = problem_match.group(1).strip() if problem_match else text

    solution_match = re.search(r'\*\*Solution:\*\*\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
    elif "Step-by-step Solution:" in text:
        solution = text.split("Step-by-step Solution:", 1)[1].strip()
    else:
        solution = "[Full solution not found]"

    explanation = ""
    lower = problem.lower()
    if any(word in lower for word in ["pizza", "slice", "pizzas"]):
        explanation = (
            "To solve this, first calculate the total number of pizza slices by multiplying the number of pizzas by the slices per pizza. "
            "Then divide that total equally among the friends or children. "
            "For juice or punch, use the given percentage or volume to find the amount per person or remaining."
        )
    elif any(word in lower for word in ["juice", "punch", "drink", "%"]):
        explanation = (
            "Identify the total amount and the percentage used or left. "
            "Convert the percentage to a decimal and multiply by the total. "
            "For sharing items, first find the total pieces, then divide equally."
        )
    else:
        explanation = (
            "Break the problem into steps: identify what needs to be shared or calculated, "
            "find totals using multiplication or addition, then divide equally or apply percentages."
        )

    return MathProblem(
        problem_statement=problem,
        solution=solution,
        explanation=explanation
    )

def extract_key_numbers(text: str) -> dict:
    numbers = {
        'pizzas': re.search(r'(\d+)\s+(?:large\s+)?pizzas?', text, re.IGNORECASE),
        'slices_per_pizza': re.search(r'(?:cut into|divided into)\s+(\d+)\s+slices', text, re.IGNORECASE),
        'children': re.search(r'(\d+)\s+children', text, re.IGNORECASE),
        'friends': re.search(r'(\d+)\s+friends', text, re.IGNORECASE),
        'percentage': re.search(r'(\d+)%', text),
        'volume_ml': re.search(r'(\d+)\s*ml', text, re.IGNORECASE),
        'volume_liters': re.search(r'(\d+)\s*liters?', text, re.IGNORECASE),
        'jugs': re.search(r'(\d+)\s+(?:large\s+)?jugs?', text, re.IGNORECASE),
    }
    return {k: int(v.group(1)) if v else None for k, v in numbers.items()}

# --- 3. THE MODELS ---
m1_model = OpenAIChatModel(
    MODEL_CONFIG['math_generator']['model'],
    provider=OpenAIProvider(base_url=MODEL_CONFIG['math_generator']['base_url'])
)
m2_model = OpenAIChatModel(
    MODEL_CONFIG['auditor']['model'],
    provider=OpenAIProvider(base_url=MODEL_CONFIG['auditor']['base_url'])
)
m3_model = OpenAIChatModel(
    MODEL_CONFIG['storyteller']['model'],
    provider=OpenAIProvider(base_url=MODEL_CONFIG['storyteller']['base_url'])
)

settings = ModelSettings(max_tokens=3072, temperature=0.7)
story_settings = ModelSettings(max_tokens=3072, temperature=0.6)

logic_gen = Agent(m1_model, output_type=str, model_settings=settings)
story_wrapper = Agent(m3_model, output_type=str, model_settings=story_settings)
auditor = Agent(m2_model, output_type=str, model_settings=settings)

# --- 4. THE PIPELINE ---
async def run_challenge_pipeline(topic: str, subDiff_level: str = '1', max_retries: int = 3, verbose: bool = False, simple_mode: bool = False):
    if subDiff_level not in SUBDIFF_LEVELS:
        console.print(f"[red]Invalid sub-difficulty level '{subDiff_level}'. Using '1' (Easy)[/red]")
        subDiff_level = '1'
    diff_info = SUBDIFF_LEVELS[subDiff_level]

    console.print(f"\n[bold cyan]🎯 Difficulty:[/bold cyan] Grade 6 - Level {subDiff_level} ({diff_info['name']})")
    console.print(f"[dim]{diff_info['description']}[/dim]\n")

    # STEP 1: Generate core math problem
    console.print(f"\n[bold yellow]🌀 Machine 1 (MathHermes-2.5):[/bold yellow] Generating math core...")
    difficulty_prompt = (
        f"Create a grade-6 level math problem about {topic}.\n"
        f"DIFFICULTY LEVEL: {diff_info['name']} (Level {subDiff_level}/3)\n"
        f"CONSTRAINTS: {diff_info['constraints']}\n\n"
        f"The final Problem Statement must be concise: no more than 120 words.\n"
        f"Avoid ambiguous phrasing. Specify units clearly.\n"
        f"Output exactly two sections:\n"
        f"Problem Statement: [the word problem]\n\n"
        f"Step-by-step Solution: [full solution with all steps]\n"
        f"Be precise and mathematically correct."
    )
    core_res = await logic_gen.run(difficulty_prompt)
    core_text = get_text_from_result(core_res)

    if "Step-by-step Solution:" not in core_text:
        console.print("[red]⚠️ Core missing solution. Regenerating...[/red]")
        core_res = await logic_gen.run(difficulty_prompt + "\nCRITICAL: Include 'Step-by-step Solution:' section.")
        core_text = get_text_from_result(core_res)

    core_numbers = extract_key_numbers(core_text)
    if verbose:
        console.print(f"[dim]Extracted numbers: {core_numbers}[/dim]")

    console.print(Panel(Markdown(core_text[:600] + ("..." if len(core_text) > 600 else "")),
                        title="[dim]Core Math Generated[/dim]", border_style="dim yellow"))

    # Split core
    if "Step-by-step Solution:" in core_text:
        core_problem_part, core_solution_part = core_text.split("Step-by-step Solution:", 1)
        core_solution = "Step-by-step Solution:" + core_solution_part.strip()
        core_problem = core_problem_part.strip()
        if "Problem Statement:" in core_problem:
            core_problem = core_problem.split("Problem Statement:", 1)[1].strip()
    else:
        core_problem = core_text.strip()
        core_solution = "[No separate solution found]"

    # STEP 2: Story enhancement (only enhance problem, preserve solution exactly)
    story_attempt = 0
    enhanced_problem_text = None
    audit_text = "No audit performed"

    if simple_mode:
        console.print(f"\n[bold magenta]🌀 Machine 3 (Llama-3.1-8B):[/bold magenta] Simple mode - skipping story enhancement")
        enhanced_problem_text = core_problem
        audit_text = "PASSED (simple mode)"
    else:
        while story_attempt < max_retries and enhanced_problem_text is None:
            story_attempt += 1
            console.print(f"\n[bold magenta]🌀 Machine 3 (Llama-3.1-8B):[/bold magenta] Enhancing problem statement (attempt {story_attempt}/{max_retries})...")

            all_numbers = [str(v) for v in core_numbers.values() if v is not None]
            numbers_list = ", ".join(all_numbers) if all_numbers else "none detected"

            story_prompt = (
                "You are turning a plain math word problem into an engaging story for students.\n"
                "Your ONLY task is to rewrite the Problem Statement with rich, descriptive details.\n\n"
                f"CRITICAL: Preserve these numbers EXACTLY: {numbers_list}\n"
                "The final Problem Statement must be no more than 120 words.\n\n"
                "You may add: character names, locations, times, weather, colors, excitement, sensory details.\n"
                "You MUST NOT: change/add/remove any numbers, perform calculations, or reveal answers.\n\n"
                "OUTPUT EXACTLY THIS FORMAT:\n"
                "**Problem Statement:**\n"
                "[Your enhanced version - maximum 120 words]\n\n"
                "DO NOT include any Solution section.\n\n"
                f"ORIGINAL PROBLEM TO ENHANCE:\n{core_problem}"
            )

            story_res = await story_wrapper.run(story_prompt)
            story_output = get_text_from_result(story_res)

            # Extract enhanced problem
            match = re.search(r'\*\*Problem Statement:\*\*\s*(.*?)(?=\*\*|$)'.strip(), story_output, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
            else:
                candidate = story_output.strip()

            word_count = count_words(candidate)
            if word_count > 120:
                console.print(f"[yellow]⚠️ Enhanced problem too long ({word_count} words). Retrying...[/yellow]")
                if story_attempt < max_retries:
                    continue

            # Number consistency check
            story_numbers = extract_key_numbers(candidate)
            mismatches = [f"{k}: {core_numbers[k]} → {story_numbers.get(k)}"
                          for k in core_numbers if core_numbers[k] and story_numbers.get(k) != core_numbers[k]]

            if mismatches:
                console.print(f"[yellow]⚠️ Number mismatch: {', '.join(mismatches)}[/yellow]")
                if story_attempt < max_retries:
                    console.print("[yellow]Retrying...[/yellow]")
                continue

            # Light audit for leakage
            console.print(f"\n[bold cyan]🌀 Machine 2 (Qwen):[/bold cyan] Quick audit...")
            audit_prompt = (
                f"Does this enhanced problem contain any calculations, totals, or solution hints?\n"
                f"Word count: {word_count} (must ≤120)\n"
                f"ENHANCED:\n{candidate}\n\n"
                f"Reply ONLY 'PASSED' or 'FAILED: [reason]'"
            )
            audit_res = await auditor.run(audit_prompt)
            audit_text = get_text_from_result(audit_res).strip()

            if "PASSED" in audit_text.upper():
                enhanced_problem_text = candidate
                console.print(f"[green]✓ Enhancement successful ({word_count} words)[/green]")
            else:
                console.print(f"[yellow]✗ Audit failed: {audit_text}[/yellow]")
                if story_attempt == max_retries:
                    console.print("[red]Max retries reached. Using original problem.[/red]")

    # Final assembly
    if enhanced_problem_text and "PASSED" in audit_text.upper():
        final_text = f"**Problem Statement:**\n{enhanced_problem_text}\n\n**Solution:**\n{core_solution}"
        used_version = "story"
    else:
        final_text = f"**Problem Statement:**\n{core_problem}\n\n**Solution:**\n{core_solution}"
        used_version = "original"
        if enhanced_problem_text is None:
            console.print("[red]Story enhancement failed. Using clean core problem.[/red]")

    problem = parse_text_problem(final_text)

    # --- DISPLAY ---
    console.print("\n" + "="*80 + "\n")
    final_word_count = count_words(problem.problem_statement)
    stats = f"Level {subDiff_level} ({diff_info['name']}) | Words: {final_word_count}/120 | Attempts: {story_attempt}/{max_retries} | Version: {used_version}"
    console.print(f"[dim]{stats}[/dim]\n")

    title = f"[bold white]📚 Student Challenge: Extract the Math (Level {subDiff_level})[/bold white]" if used_version == "story" else f"[bold yellow]📚 Original Problem (Story Failed)[/bold yellow]"
    console.print(Panel(Markdown(problem.problem_statement),
                        title=title,
                        border_style="magenta" if used_version == "story" else "yellow",
                        padding=(1, 2)))

    if problem.explanation.strip():
        console.print(Panel(Markdown(problem.explanation),
                            title="[bold blue]🧭 How to Solve It[/bold blue]",
                            border_style="bright_blue",
                            padding=(1, 2)))

    console.print(Panel(Markdown(problem.solution),
                        title="[bold white]✅ Teacher's Solution Key[/bold white]",
                        border_style="green",
                        padding=(1, 2)))

    audit_status = "✓ PASSED" if "PASSED" in audit_text.upper() else "✗ ISSUES"
    console.print(Panel(Markdown(audit_text),
                        title=f"[bold white]🔍 Qwen Audit: {audit_status}[/bold white]",
                        border_style="cyan",
                        subtitle=f"Version: {used_version} | Numbers: {core_numbers}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triple-Agent Math Problem Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                              # Default topic + Level 1
  python script.py -l 2                         # Level 2
  python script.py --level 3 "pirates dividing treasure"
  python script.py -v -s                         # Verbose + simple mode
        """.strip()
    )
    parser.add_argument('topic', nargs='*', default=None, help="Custom topic (in quotes if multi-word)")
    parser.add_argument('-l', '--level', type=str, choices=['1','2','3'], default='1', help="Difficulty: 1=Easy, 2=Medium, 3=Difficult")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--simple', action='store_true', help="Minimal story changes")

    args = parser.parse_args()

    topic = " ".join(args.topic) if args.topic else "a birthday party where they need to share pizza (fractions) and juice (percentages)"
    subDiff_level = args.level
    verbose = args.verbose
    simple_mode = args.simple

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan] Triple-Agent Math Problem Generator Pipeline[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    console.print("[bold]🤖 Model Configuration:[/bold]")
    for role, config in MODEL_CONFIG.items():
        ctx = config.get('max_context', 'Unknown')
        console.print(f" [yellow]•[/yellow] [bold]{config['name']}[/bold]: {config['role']} [dim](ctx: {ctx})[/dim]")
    console.print()

    console.print("[bold]🎯 Available Sub-Difficulty Levels:[/bold]")
    for level, info in SUBDIFF_LEVELS.items():
        console.print(f" [yellow]Level {level}[/yellow] - [bold]{info['name']}[/bold]: {info['description']}")
    console.print()

    console.print(f"[bold blue]📝 Topic:[/bold blue] {topic}")
    console.print(f"[bold cyan]🎯 Difficulty Level:[/bold cyan] {subDiff_level} ({SUBDIFF_LEVELS[subDiff_level]['name']})")
    if verbose: console.print("[dim]Verbose mode enabled[/dim]")
    if simple_mode: console.print("[dim]Simple mode enabled[/dim]")
    console.print()

    asyncio.run(run_challenge_pipeline(
        topic=topic,
        subDiff_level=subDiff_level,
        verbose=verbose,
        simple_mode=simple_mode
    ))