"""Discover available MSMARCO-XI dataset splits and sizes."""
import sys


def discover():
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder("unicamp-dl/mmarco", "portuguese")
        print("Dataset builder loaded:", builder.info.description[:200])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    discover()
