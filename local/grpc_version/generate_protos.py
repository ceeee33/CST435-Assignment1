import sys
from grpc_tools import protoc


def main() -> int:
    args = [
        "protoc",
        "-I.",
        "--python_out=.",
        "--grpc_python_out=.",
        "pipeline.proto",
    ]
    return 0 if protoc.main(args) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


