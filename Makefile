ifeq ($(OS),Windows_NT) 
    detected_OS := Windows
	SYSTEM_PYTHON  = $(or $(where.exe python3), $(where.exe python))
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
	SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
endif

create-patch-version:
	@echo "Creating patch version"
	@bump2version --no-commit --current-version 0.1.2 dev src/python/VERSION