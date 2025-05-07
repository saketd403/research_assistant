import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Query ChatGPT API')

    parser.add_argument('--topic', type=str, default="Best way to learn AI",
                        help='Enter your research topic')
    parser.add_argument('--max_analysts', type=str, default=3,
                        help='Maximum number of interviewers/analysts.')
    args = parser.parse_args()

    return args