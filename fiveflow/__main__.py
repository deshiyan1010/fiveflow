from .utils import check_for_updates, request_accessibility, prompt_hf_token_if_needed
from .pipeline import Pipeline


def main():
    check_for_updates()
    request_accessibility()
    prompt_hf_token_if_needed()
    Pipeline().run()


if __name__ == "__main__":
    main()
