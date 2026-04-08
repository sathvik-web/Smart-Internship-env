from env.environment import InternshipEnv


def main() -> InternshipEnv:
    """OpenEnv server entrypoint.

    Returns a fresh environment instance for the server runtime.
    """
    return InternshipEnv()


if __name__ == "__main__":
    main()
