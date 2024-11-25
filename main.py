def evaluate_object(obj):
    if isinstance(obj, (int, float, bool)):
        # If it's an int, float, or bool, return it as is
        return obj

    elif isinstance(obj, str):
        # Try evaluating the string, if it fails treat it as plain text
        try:
            # Attempt evaluation
            evaluated = eval(obj)
            return evaluated

        except (SyntaxError, NameError):
            # Return as plain string if eval fails with SyntaxError or NameError
            return obj
    else:
        # For other types, you might want to return a string representation
        return repr(obj)


def parse_args_to_dict(args):
    """Convert list of arguments to a dictionary, allowing multiple values for the same key."""
    kwargs = {}
    it = iter(args)

    for arg in it:
        if arg.startswith("--"):
            # Handle long option names
            key = arg[2:]  # Remove the leading '--'
            value = next(it, None)
            if value is not None:
                if key in kwargs:
                    # If the key already exists, append the new value to the list
                    if isinstance(kwargs[key], list):
                        kwargs[key].append(evaluate_object(value))
                    else:
                        kwargs[key] = [kwargs[key], evaluate_object(value)]
                else:
                    kwargs[key] = evaluate_object(value)  # First occurrence

        elif arg.startswith("-"):
            # Handle short option names
            key = arg[1:]  # Remove the leading '-'
            value = next(it, None)
            if value is not None:
                if key in kwargs:
                    # If the key already exists, append the new value to the list
                    if isinstance(kwargs[key], list):
                        kwargs[key].append(evaluate_object(value))
                    else:
                        kwargs[key] = [kwargs[key], evaluate_object(value)]
                else:
                    kwargs[key] = evaluate_object(value)  # First occurrence
    return kwargs


def main():
    import sys

    # Get command-line arguments
    args = sys.argv[1:]  # Skip the script name

    # Check for the env argument
    if "-e" in args or "--env" in args:
        dotenv_index = args.index("-e") if "-e" in args else args.index("--env")
        dotenv_path = args[dotenv_index + 1]  # The next argument is the env path

        # Remove the env path and its argument from args
        args.pop(dotenv_index)  # Remove '-e' or '--env'
        args.pop(dotenv_index)  # Remove the env path itself

    else:
        dotenv_path = "./.env"

    # Load env
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=dotenv_path)

    from main_methods import (
        module_dict,
    )  # Import the dictionary containing your module functions

    # Check for the required module key argument
    if "-m" not in args and "--module-key" not in args:
        print("Error: The module key argument is required (-m or --module-key).")
        exit(1)

    # Extract the module key
    try:
        module_key_index = (
            args.index("-m") if "-m" in args else args.index("--module-key")
        )
        module_key = args[module_key_index + 1]  # The next argument is the module key
    except (ValueError, IndexError):
        print("Error: No module key provided after '-m' or '--module-key'.")
        exit(1)

    # Remove the module key and its argument from args
    args.pop(module_key_index)  # Remove '-m' or '--module-key'
    args.pop(module_key_index)  # Remove the module key itself

    # Parse remaining arguments into a dictionary
    module_kwargs = parse_args_to_dict(args)

    # Check if the provided key exists in module_dict
    if module_key in module_dict:
        # Call the main function of the corresponding module with additional keyword arguments
        module_dict[module_key](**module_kwargs)  # Unpack named arguments
    else:
        print(f"Error: No module found for key '{module_key}'")
        print("\nAvailable module keys:")
        for key in module_dict.keys():
            print(f"  - {key}")
        exit(1)


if __name__ == "__main__":
    main()
