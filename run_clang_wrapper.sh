#!/bin/bash
SDK="/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk"
INC="$SDK/usr/include/c++/v1"
exec /usr/bin/clang++ -I"$INC" -isysroot "$SDK" "$@"
