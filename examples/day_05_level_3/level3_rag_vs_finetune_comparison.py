from dataclasses import dataclass


@dataclass
class Scenario:
	has_frequent_knowledge_updates: bool
	needs_strong_style_control: bool
	has_small_budget: bool


def choose_strategy(scenario: Scenario) -> str:
	if scenario.has_frequent_knowledge_updates:
		return "RAG"
	if scenario.needs_strong_style_control and not scenario.has_small_budget:
		return "Fine-tuning"
	return "RAG + prompt engineering"


def main() -> None:
	print("RAG vs Fine-tuning quick guide")
	print("- RAG: best when data changes often and sources must be cited")
	print("- Fine-tuning: best when behavior/style must be deeply customized")
	print("- Hybrid: use RAG for freshness, tune prompts/model for tone")

	sample = Scenario(
		has_frequent_knowledge_updates=True,
		needs_strong_style_control=False,
		has_small_budget=True,
	)
	print(f"\nRecommended strategy for sample scenario: {choose_strategy(sample)}")


if __name__ == "__main__":
	main()

