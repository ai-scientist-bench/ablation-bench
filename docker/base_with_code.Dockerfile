ARG NAME
FROM ablations-bench:${NAME}

ARG GITHUB_URL
ARG COMMIT_SHA

RUN rm -rf /repo
RUN git clone ${GITHUB_URL} repo
RUN cd /repo && git reset --hard ${COMMIT_SHA}

SHELL ["/bin/bash", "-c"]