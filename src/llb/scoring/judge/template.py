"""Ukrainian prompt templates for DeepEval G-Eval metrics."""

from llb.prompts import render_text, render_text_list, render_text_map

UA_FAITHFULNESS_STEPS = render_text_list("scoring.judge.faithfulness_steps")
UA_ANSWER_RELEVANCY_STEPS = render_text_list("scoring.judge.relevancy_steps")


class UkrainianGEvalTemplate:
    """DeepEval G-Eval result prompt with Ukrainian-only judge instructions."""

    _PARAMETER_LABELS = render_text_map("scoring.judge.parameter_labels")

    @classmethod
    def _localize_parameter_labels(cls, text: str) -> str:
        for english, ukrainian in cls._PARAMETER_LABELS.items():
            text = text.replace(english, ukrainian)
        return text

    @classmethod
    def generate_evaluation_results(
        cls,
        evaluation_steps: str,
        test_case_content: str,
        parameters: str,
        rubric: str | None = None,
        score_range: tuple[int, int] = (0, 10),
        _additional_context: str | None = None,
        multimodal: bool = False,
    ) -> str:
        del rubric, _additional_context, multimodal
        return render_text(
            "scoring.judge.evaluation_results",
            {
                "score_min": score_range[0],
                "score_max": score_range[1],
                "evaluation_steps": evaluation_steps,
                "test_case_content": cls._localize_parameter_labels(test_case_content),
                "parameters": cls._localize_parameter_labels(parameters),
            },
        )

    @classmethod
    def generate_strict_evaluation_results(
        cls,
        evaluation_steps: str,
        test_case_content: str,
        parameters: str,
        _additional_context: str | None = None,
        multimodal: bool = False,
    ) -> str:
        return cls.generate_evaluation_results(
            evaluation_steps,
            test_case_content,
            parameters,
            score_range=(0, 1),
            _additional_context=_additional_context,
            multimodal=multimodal,
        )
