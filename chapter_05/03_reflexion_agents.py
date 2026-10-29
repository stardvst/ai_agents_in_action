import asyncio

from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel


@function_tool
def check_answer(days: int | None) -> bool:
    """Return whether the proposed number of days is correct."""
    return days == 26


class CriticResult(BaseModel):
    correct: bool
    hint: str


def get_reflexion_solver_instructions(
    run_context: RunContextWrapper[str], agent: Agent[str]
) -> str:
    """Generate instructions for the reflexion solver agent."""
    instructions = (
        "You are a time-travel expert. Solve the problem step by step "
        "and be careful to avoid mistakes."
    )
    return instructions + "\nHINT:\n" + run_context.context


# --- Base agents -------------------------------------------------------------
solver = Agent(
    name="TimeTravelerReflexion", instructions=get_reflexion_solver_instructions
)
critic = Agent(
    name="TimeTravelCritic",
    instructions=(
        "You are an expert tutor. If the solution is wrong, explain the error "
        "and give a concise hint for improvement."
    ),
    tools=[check_answer],
    output_type=CriticResult,
)

# --- Problem spec ------------------------------------------------------------
problem = (
    "I left the year 2000 in a time machine, went forward 30 years, "
    "then back 40 years. I claim I'm now in 1990. Am I correct?"
)
problem = """
In a sci-fi film, Alex is a time traveler who decides to go back in time
to witness a famous historical event that took place 100 years ago,
which lasted for 10 days. He arrives three days before the event starts.
However, after spending six days in the past, he jumps forward in time
by 50 years and stays there for 20 days. Then, he travels back to
witness the end of the event. 
How many days does Alex spend in the past before he sees the end of the event?
"""
TARGET_DAYS = "26"  # expected final answer
MAX_ATTEMPTS = 4  # fail-safe cap on retries


async def main():
    # --- Reflexion loop ----------------------------------------------------------
    feedback_hint = ""
    for attempt_no in range(1, MAX_ATTEMPTS + 1):
        # Run the solver
        result = await Runner.run(solver, input=problem, context=feedback_hint)
        answer = result.final_output.strip()
        print(f"\nAttempt {attempt_no}:\n{answer}")

        # --- Simple correctness check -------------------------------------------
        has_correct_days = TARGET_DAYS in answer
        says_claim_correct = "yes" in answer.lower() or "correct" in answer.lower()
        solved = has_correct_days and says_claim_correct

        if solved:
            print("✅ Solution accepted.")
            break

        # --- Not solved: generate feedback & retry ------------------------------
        feedback_prompt = (
            f"Problem:\n{problem}\n\n"
            f"Solution given:\n{answer}\n\n"
            "Check the final number of days and provide feedback."
        )

        feedback_resp = await Runner.run(critic, input=feedback_prompt)
        review = feedback_resp.final_output

        if review.correct:
            print("✅ Solution accepted.")
            break

        print(f"Feedback:\n{review.hint}")
        feedback_hint = review.hint
    else:
        print("\n⚠️  Max attempts reached without a correct solution.")


if __name__ == "__main__":
    asyncio.run(main())
