## DSPSR

[![pipeline status](https://gitlab.com/ska-telescope/dspsr/badges/master/pipeline.svg)](https://gitlab.com/ska-telescope/dspsr/commits/master)

### Documentation

Documentation for DSPSR can be found online [here](http://dspsr.sourceforge.net/).

### Building

DSPSR is dependent on [PSRCHIVE](https://sourceforge.net/projects/psrchive/). 
It has several other dependencies, but the build system will direct you to install them if it detects that they aren't present.

With autotools installed, building can be as simple as the following:

```bash
./bootstrap
mkdir build && cd build
./../configure
make
make install
```

### Containerised development environment

#### BASE builder

The base builder contains curated preinstalled dependencies. Details located at [ska-pst-dsp-tools](https://gitlab.com/ska-telescope/pst/ska-pst-dsp-tools)

```bash
#!/bin/bash
# Set env vars
DOCKER_IMAGE=registry.gitlab.com/ska-telescope/pst/ska-pst-dsp-tools/ska-pst-dspsr-builder:0.2.0

# Pull the container image
docker pull $DOCKER_IMAGE
```

#### Optional: Install missing dependencies

The following are example steps for installing desired dspsr dependencies.

```bash
#!/bin/bash
# launch a base builder container
docker run -tid --name dspsr_$(whoami) \
-v $PWD:/home/pst/src/dspsr \
-w /home/pst/src/dspsr \
-u root \
$DOCKER_IMAGE

# launch a shell session inside the base builder's filesystem
docker exec -ti dspsr_$(whoami) bash

# Install sample desired dependency: PSRDADA
git clone --recursive https://git.code.sf.net/p/psrdada/code /home/pst/src/psrdada && cd /home/pst/src/psrdada
chmod +x ./bootstrap && ./bootstrap
mkdir -p /home/pst/build/psrdada && cd /home/pst/build/psrdada
/home/pst/src/psrdada/configure \
    --with-cuda-include-dir=/usr/local/cuda/include \
    --with-cuda-lib-dir=/usr/local/cuda/lib64 \
    --prefix=/home/pst --enable-shared \
    && make -j$(nproc) && make install

# Post installation verification
which dada_db && dada_db -h

# Prepare dspsr compile configuration and compile
cd /home/pst/src/dspsr
./bootstrap
mkdir -p /home/pst/build/dspsr && cd /home/pst/build/dspsr
echo 'uwb dada sigproc dummy fits vdif ska1 cpsr2' > ./backends.list
/home/pst/src/dspsr/configure \
    --with-cuda-include-dir=/usr/local/cuda/include \
    --with-cuda-lib-dir=/usr/local/cuda/lib64 \
    --prefix=/home/pst --enable-shared \
    && make -j$(nproc)
```

### GTest unit tests

[GoogleTest](https://google.github.io/googletest/) Unit testing framework can be built and executed using
a premade docker image containing curated dspsr dependencies. Below are the command line instructions
for pulling the involved docker image used for building dspsr and launching googletest unit tests.

```bash
#!/bin/bash
# launch a shell session against running dspsr container with pre installed desired dependencies
# Set env vars
DOCKER_IMAGE=registry.gitlab.com/ska-telescope/pst/ska-pst-dsp-tools/ska-pst-dspsr-builder:0.2.0

# Pull the container image
docker pull $DOCKER_IMAGE

# Launch a detached docker container with root as the user id within the container
docker run -tid --name dspsr_unittest \
-v $PWD:/home/pst/src/dspsr \
-w /home/pst/src/dspsr \
-u root \
$DOCKER_IMAGE

# Launch a shell that access the container's file system
docker exec -ti dspsr_unittest bash

# cd to build directory to build and execute gtest unit tests that have been configured
cd /home/pst/build/dspsr && make check
```
