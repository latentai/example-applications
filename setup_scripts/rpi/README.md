# Example Application - Dependency Install Scripts for Raspberry Pi 4B
This directory includes a script for setting up Raspberry Pi 4B devices to use the example applications in this repo.

The following are the recommended steps to using these install scripts:

1. Start with a fresh install of the OS on your Raspberry Pi.  We currently only support *Raspberry Pi Bullseye 64-bit OS (Debian 11)*
2. Add the Latent AI apt server:<br>
`wget -qO - https://public.latentai.io/add_apt_repository | sudo bash`
3. Run the install script:<br>
`install-for-bullseye.sh`
4. Install the LOR Server (`leip-lor`) as instructed in the [LEIP Documentation](https://leipdocs.latentai.io/cf/3.0/content/modules/lor/) if you are intending to run the `lor_server` for remote use of *leip evaluate*<br>


## Customizing this process
The above install script is intended as an example.  It is suggested that you
follow the above install process during your evaluation phase with LEIP Recipes.


A number of these dependencies are required for the LOR and example pre & post processing. Your dependencies will vary.
In a deployment scenario, your dependency list will likely be much shorter, particularly if you are deploying with C++
applications.  At a minimum, you will require the following packages:

1. `liblre-cpu`
2. `liblre-dev`
3. Additional dependencies required by your preprocessing, postprocessing and application

