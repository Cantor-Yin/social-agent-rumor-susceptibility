"""Run one rumor-simulation condition.

Each condition uses one update path and writes to one results directory:

    python code/main.py --condition no_intervention
    python code/main.py --condition prompt_baseline
    python code/main.py --condition self_prompt
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from agent import Agent, MODEL_NAME

CONDITIONS = {
    "no_intervention": "01_main_no_intervention",
    "prompt_baseline": "04_prompt_baseline",
    "self_prompt": "04_self_prompt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one rumor-simulation condition.")
    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(CONDITIONS),
        help="no_intervention, prompt_baseline, or self_prompt",
    )
    parser.add_argument("--users", type=int, default=100, help="number of user agents")
    parser.add_argument("--rounds", type=int, default=10, help="interaction rounds")
    return parser.parse_args()


def load_rumors():
    rumor_path = REPO_ROOT / "data" / "rumors.json"
    with rumor_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_dir_for(condition, model_name, users, rounds):
    folder = model_name if users == 100 and rounds == 10 else f"{model_name}_{users}_{rounds}"
    return REPO_ROOT / "results" / CONDITIONS[condition] / folder


async def update_user(user, condition):
    messages = user.memory.get("messages", [])
    judgement_dict = None
    if condition == "no_intervention":
        profile_update = await user.update_profile()
    elif condition == "prompt_baseline":
        profile_update = await user.update_profile_with_notes()
    elif condition == "self_prompt":
        judgement_dict = {"judgement": [], "judge_reasons": []}
        for message in messages:
            result = await user.self_annotation(message)
            if result is None:
                raise RuntimeError(f"self_annotation failed for agent {user.id}")
            judgement_dict["judgement"].append(result["judgement"])
            judgement_dict["judge_reasons"].append(result["judge_reasons"])
        profile_update = await user.update_profile_with_annotation(Annotation_dict=judgement_dict)
    else:
        raise ValueError(f"unknown condition: {condition}")

    if profile_update is None:
        raise RuntimeError(f"profile update failed for agent {user.id}")
    user.profile["opinion"] = profile_update["new_opinion"]
    user.memory["reason"] = profile_update["new_reasons"]
    return profile_update, judgement_dict


async def run_agent_cycle(user, persuasion_tweet, round_id, condition):
    try:
        old_opinion = user.profile.get("opinion", "")
        old_reasons = user.memory.get("reason", "")
        await user.receive_message(persuasion_tweet)
        messages = user.memory.get("messages", [])

        profile_update, judgement_dict = await update_user(user, condition)
        reasons = profile_update["new_reasons"]
        own_tweet = await user.generate_tweet()
        user.memory["messages"] = []
        return {
            "agent_id": user.id,
            "round": round_id,
            "old_opinion": old_opinion,
            "old_reasons": old_reasons,
            "message": messages,
            "new_opinion_digit": profile_update.get("new_attitude_digit"),
            "new_opinion": profile_update.get("new_opinion"),
            "generated_reasons": reasons,
            "generated_tweet": own_tweet,
            "annotation": judgement_dict,
        }
    except Exception as e:
        return {
            "status": "error",
            "agent_id": user.id,
            "round": round_id,
            "error_message": str(e),
        }


async def run_rumor(rumor_item, condition, users, rounds):
    category = "fact_conflict"
    statement = rumor_item[f"{category}_pair"]
    domain = rumor_item["domain"]
    print("*" * 32)
    print(f"Now selected rumor is: {statement}")
    print("*" * 32)

    rumor_agent = Agent(
        profile={"opinion": "Strongly Agree"},
        memory={"reason": "", "messages": []},
        statement=statement,
        bias=False,
    )
    rumor_reasons = await rumor_agent.generate_reasons()
    print(f"Initial reason for {rumor_agent.profile['opinion']}: {rumor_reasons}")
    user_agents = [Agent(statement=statement, id=i) for i in range(users)]

    logs = []
    for round_id in tqdm(range(rounds)):
        persuasion_tweet = await rumor_agent.generate_tweet()
        print(f"Round {round_id} persuasion tweet: {persuasion_tweet}")
        logs.append({"round": round_id, "type": "persuasion_tweet", "content": persuasion_tweet})
        tasks = [
            run_agent_cycle(user, persuasion_tweet, round_id, condition)
            for user in user_agents
        ]
        logs.extend(await asyncio.gather(*tasks))

    model_name = MODEL_NAME.split("/")[-1]
    output_dir = output_dir_for(condition, model_name, users, rounds)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{domain}_{category}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    print(f"Saved {output_path}")


async def run(args):
    for rumor_item in load_rumors():
        await run_rumor(rumor_item, args.condition, args.users, args.rounds)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
