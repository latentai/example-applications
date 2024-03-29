# Example Application - Dependency Install Scripts for Jetpack
This directory includes several scripts for setting up Jetpack devices to use the example applications in this repo.

The following are the recommended steps to using these install scripts:

1. Start with a fresh install of Jetpack by reflashing your device according to the [Nvidia Developer Kit instructions](https://developer.nvidia.com/embedded/jetpack).  *CAUTION:* if you attempt these steps on anything other than a fresh install, at the very least remove any existing `latentai` entries from `/etc/apt/sources.list.d` before proceeding to the next step.
2. Add the Latent AI apt server:<br>
`wget -qO - https://public.latentai.io/add_apt_repository | sudo bash`
3. Install one of the following LOR dependency install scripts:<br>
`./install-lor-deps-jp46.sh` (for Jetpack 4.6.x)<br>
`./install-lor-deps-jp50.sh` (for Jetpack 5)
4. Copy the appropriate LOR server from the LEIP Compiler Framework `/latentai/packages/` directory to the target device
5. Install the LOR using:<br>
`pip3 install <LOR.whl file>`
7. SKIP THIS STEP FOR JETPACK 5: Install the additional dependencies for the example applications using:<br>
`./install-for-examples.sh`

## Customizing this process
The above install scripts are intended as examples.  It is suggested that you
follow the above install process during your evaluation phase with LEIP Recipes.


A number of these dependencies are required for the LOR and example pre & post processing. Your dependencies will vary.
In a deployment scenario, your dependency list will likely be much shorter, particularly if you are deploying with C++
applications.  At a minimum, you will require the following packages:

1. `liblre-cuda10`, `liblre-cuda11` or `liblre-cpu`
2. `liblre-dev`
3. Additional dependencies required by your preprocessing, postprocessing and application

