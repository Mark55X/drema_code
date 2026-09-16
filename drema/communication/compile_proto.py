#!/usr/bin/env python
"""
Utility script to compile drema_comm.proto into Python gRPC stubs.
"""
import os
import sys
from grpc_tools import protoc

def compile_proto():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    proto_dir = os.path.join(curr_dir, "proto")
    proto_file = os.path.join(proto_dir, "drema_comm.proto")
    out_dir = proto_dir

    print(f"Compiling {proto_file} -> {out_dir}...")
    args = [
        "protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        proto_file
    ]
    ret = protoc.main(args)
    if ret != 0:
        raise RuntimeError(f"protoc failed with exit code {ret}")

    # Fix relative import in generated grpc stub if needed
    grpc_stub = os.path.join(out_dir, "drema_comm_pb2_grpc.py")
    if os.path.exists(grpc_stub):
        with open(grpc_stub, "r") as f:
            content = f.read()
        # Ensure import drema_comm_pb2 works both directly and as package
        if "import drema_comm_pb2 as drema__comm__pb2" in content:
            fixed_content = content.replace(
                "import drema_comm_pb2 as drema__comm__pb2",
                "from . import drema_comm_pb2 as drema__comm__pb2"
            )
            with open(grpc_stub, "w") as f:
                f.write(fixed_content)
            print("✓ Fixed relative import in drema_comm_pb2_grpc.py")

    # Also make sure proto dir has an __init__.py exposing them
    proto_init = os.path.join(out_dir, "__init__.py")
    with open(proto_init, "w") as f:
        f.write("from . import drema_comm_pb2\nfrom . import drema_comm_pb2_grpc\n")

    print("✓ Successfully generated Protobuf and gRPC Python modules in:", out_dir)

if __name__ == "__main__":
    compile_proto()
