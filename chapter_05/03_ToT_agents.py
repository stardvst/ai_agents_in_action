import asyncio

from agents import Agent, Runner
from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    score: float = Field(ge=0, le=10)

# Agent for generating next-step thoughts
generator = Agent(
    name="ToT-Generator",
    instructions="Given the current situation, brainstorm a possible next step or action to reach the goal.",
)
# Agent for evaluating partial solutions
evaluator = Agent(
    name="ToT-Evaluator",
    instructions="Assess how likely the proposed plan will solve the problem. Respond with score = <number 0-10>.",
    output_type=Evaluation,
)

problem = """
You need to reach the year 1800 from 2025 using a time machine
that can jump either -100 or -30 years.
"""


async def main():
    # Generate initial thought candidates
    initial_thoughts = []
    for i in range(2):
        resp = await Runner.run(
            generator, input=f"Problem: {problem}\nThink of a first step."
        )
        initial_thoughts.append(resp.final_output.strip())

    # Evaluate and expand each thought (one iteration of BFS expansion)
    promising_branches = []
    for thought in initial_thoughts:
        eval_resp = await Runner.run(
            evaluator, input=f"Problem: {problem}\nPlan: {thought}\nIs this promising?"
        )
        if eval_resp.final_output.score >= 7:
            # Expand this thought with a second step
            next_step = await Runner.run(
                generator, input=f"Current idea: {thought}\nNext step?"
            )
            promising_branches.append(f"{thought} -> {next_step.final_output.strip()}")

    print("Initial thought candidates:", initial_thoughts)
    print(
        "Expanded promising branch:",
        promising_branches[0] if promising_branches else "None",
    )


if __name__ == "__main__":
    asyncio.run(main())
