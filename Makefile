all:
	clang++ -static -fuse-ld=lld -g -O2 -o what-the-FEX what-the-FEX.cpp -std=c++20 `pkgconf --libs --static ncursesw`
