"""Document decisions on the format of `stats.yml`.

stats = [{
        "model_name": str,
        "paragraph_id": str,
        "answer": {
            'parameter': answer,
            'parameter_unit': answer,
            ("actual_a": str | int,)  # optional
            ("actual_m": str | int,)  # optional

            for each parameter
        },
    },
]

"""
