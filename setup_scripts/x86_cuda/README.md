# Example Application - Dependency Install Scripts LEIP SDK
The LEIP SDK comes pre-installed with all the dependencies that you will need to run the LOR in the container.


If you would like to run the example applications in this repo within the LEIP SDK, you will need to install
some additional dependencies such as `torch` and `torchvision`.  This directory includes a script to install those dependencies
and can be run using the following:

1. Add the Latent AI apt server:<br>
`wget -qO - https://public.latentai.io/add_apt_repository | sudo bash`
2. Install the additional dependencies for the example applications using:<br>
`./install-for-examples.sh`


## Customizing this process
If you wish to install the example_applications on an x86_64 machine outside of the LEIP SDK environment, you may
need to install additional dependencies.

Your dependencies will vary. In a deployment scenario, your dependency list will likely be much shorter,
particularly if you are deploying with C++ applications.  At a minimum, you will require the following packages:

1. `liblre-cuda` or `liblre-cpu`
2. `liblre-dev`
3. Additional dependencies required by your preprocessing, postprocessing and application

