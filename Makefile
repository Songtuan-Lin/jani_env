# -------- Config --------
IMAGE := garricklin/jani_env
SHA := $(shell git rev-parse --short HEAD)
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Optional: detect dirty repo
DIRTY := $(shell git diff --quiet || echo "-dirty")

FULL_SHA := $(SHA)$(DIRTY)

# -------- Targets --------

.PHONY: build
build:
	docker build --pull \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		.

.PHONY: push
push: build
	docker push $(IMAGE):latest
	docker push $(IMAGE):$(FULL_SHA)

.PHONY: print-version
print-version:
	@echo "Git SHA: $(FULL_SHA)"
	@echo "Build time: $(BUILD_TIME)"

.PHONY: no-cache
no-cache:
	docker build --no-cache \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		.
