io_schema = {"status": 0, "stdout": [""], "stderr": [""]}
config_schema = {
    "compile": {"command": [""], "output": io_schema},
    "runtime": {"command": [""], "output": io_schema},
}


class JSONConfigError(ValueError):
    pass


def validate_config_follows_schema(config: dict, current_key="", schema=config_schema):
    for key, option in config.items():
        full_key = f"{current_key}.{key}"

        # Validate keys
        if key not in schema:
            raise JSONConfigError(f"'{full_key}' not recognized")

        # Validate types
        schema_value = schema[key]
        if type(option) is not type(schema_value):
            raise JSONConfigError(
                f"'{full_key}' must be of type '{type(schema_value)}' not '{type(option)}'"
            )

        # Validate list elements are also the correct type
        if isinstance(schema_value, list):
            for item in option:
                if type(item) is not type(schema_value[0]):
                    raise JSONConfigError(
                        f"'{full_key}' must be of type '[{type(schema_value[0])}]' "
                        f"not '[{type(item)}]'"
                    )

        # Recurse for dict types
        if isinstance(schema_value, dict):
            validate_config_follows_schema(option, full_key, schema_value)

    return True
