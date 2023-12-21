# Example Application - Dependency Install Scripts LEIP Compiler Framework
The LEIP Compiler Framework comes pre-installed with all the dependencies that you will need to run the LOR in the container.


If you would like to run the example applications in this repo within the LEIP Compiler Framework, you will need to install
some additional dependencies such as `torch` and `torchvision`.  This directory includes a script to install those dependencies
and can be run using the following:

1. Add the Latent AI apt server:<br>
`wget -qO - https://public.latentai.io/add_apt_repository | bash`<br>
`apt update`
3. Install the additional dependencies for the example applications using:<br>
`./install-for-examples.sh`


## Customizing this process
If you wish to install the example_applications on an x86_64 machine outside of the LEIP Compiler Framework environment, you may
need to install additional dependencies.

Your dependencies will vary. In a deployment scenario, your dependency list will likely be much shorter,
particularly if you are deploying with C++ applications.  At a minimum, you will require the following packages:

1. `liblre-cuda` or `liblre-cpu`
2. `liblre-dev`
3. Additional dependencies required by your preprocessing, postprocessing and application

