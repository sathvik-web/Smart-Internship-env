from __future__ import annotations

from env.models import InternshipOption, InternshipTask


def load_tasks() -> list[InternshipTask]:
	"""Return deterministic internship recommendation tasks."""
	easy_task = InternshipTask(
		task_id="easy-apply-ignore-001",
		difficulty="easy",
		objective="Classify whether the student should apply to a backend internship.",
		internship_title="Backend Engineering Intern - API Integrations",
		description=(
			"Work with the platform team to build and maintain REST endpoints for internship"
			" matching workflows. Implement endpoint tests, fix production bugs, and participate"
			" in weekly sprint planning."
		),
		required_skills=["python", "fastapi", "sql", "git", "api-design"],
		student_skills=["python", "flask", "sql", "git", "docker", "pytest"],
		correct_decision="apply",
		true_score=0.74,
		expected_reasoning_keywords=["python", "sql", "backend", "api", "fit"],
	)

	medium_task = InternshipTask(
		task_id="medium-relevance-001",
		difficulty="medium",
		objective=(
			"Provide a nuanced relevance score in [0.0, 1.0] and decision by balancing strong"
			" core matches against missing MLOps skills."
		),
		internship_title="ML Operations Intern - Recommendation Systems",
		description=(
			"Support model deployment for internship recommendation pipelines. Build feature"
			" quality checks, monitor model drift, and work with ML engineers on release"
			" readiness."
		),
		required_skills=["python", "mlops", "docker", "airflow", "monitoring", "sql"],
		student_skills=["python", "sql", "docker", "pytorch", "linux", "git"],
		correct_decision="apply",
		true_score=0.61,
		expected_reasoning_keywords=["mlops", "docker", "monitoring", "gaps", "pipeline"],
	)

	hard_task = InternshipTask(
		task_id="hard-ranking-001",
		difficulty="hard",
		objective=(
			"Rank four internships from best to worst fit, decide apply/ignore for the top-ranked"
			" role, and output a calibrated relevance score for the top-ranked role."
		),
		internship_title="Cross-Domain Internship Portfolio Ranking",
		description=(
			"Prioritize opportunities across ML research, backend systems, analytics, and"
			" frontend engineering. Ranking quality is evaluated using pairwise ordering"
			" consistency with a deterministic gold ranking."
		),
		required_skills=["skill-matching", "reasoning", "ranking", "tradeoff-analysis"],
		student_skills=["python", "sql", "pytorch", "nlp", "docker", "git", "fastapi"],
		correct_decision="apply",
		true_score=0.81,
		expected_reasoning_keywords=["ranking", "fit", "ml", "backend", "tradeoff"],
		internship_options=[
			InternshipOption(
				internship_title="Applied NLP Research Intern",
				description=(
					"Build and evaluate transformer models for resume parsing and recommendation"
					" ranking tasks."
				),
				required_skills=["python", "pytorch", "nlp", "ml-evaluation"],
			),
			InternshipOption(
				internship_title="Backend Platform Intern",
				description=(
					"Develop FastAPI microservices, optimize SQL queries, and maintain CI for"
					" internship workflow systems."
				),
				required_skills=["python", "fastapi", "sql", "docker"],
			),
			InternshipOption(
				internship_title="Data Analytics Intern",
				description=(
					"Design dashboards for recruiter funnel metrics and perform cohort analysis"
					" for engagement trends."
				),
				required_skills=["sql", "statistics", "bi", "python"],
			),
			InternshipOption(
				internship_title="Frontend Product Intern",
				description="Build recruiter-facing UI components using React and TypeScript.",
				required_skills=["react", "typescript", "css", "testing"],
			),
		],
		correct_ranking=[
			"Applied NLP Research Intern",
			"Backend Platform Intern",
			"Data Analytics Intern",
			"Frontend Product Intern",
		],
	)

	return [easy_task, medium_task, hard_task]

