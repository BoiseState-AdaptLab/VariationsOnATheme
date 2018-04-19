# VIBE
This repo contains the source code for the VIBE project.
## Setup Instructions
### Pre-requisites
You will need to install the following items:
  1. GCC X, where X > 5 as of Jan. 2016
    * The version of CUDA being used will determine which GCC is acceptable
  2. OpenCV X (X >=2)
  3. GStreamer (or similar-purpose Video streaming software)

Please feel free to refer to **Setup Walkthrough** section to read the exact steps I took when I did this.

### Steps
  1. make
If you'd like to build a specific part of the project (i.e. NaiveSerial-XXX), then you can do so by using an appropriate ```make``` rule.
Refer to ```Makefile``` for list of supported rules.

  **Note** In case of build errors, make sure you fix value of ```CXX``` variable in the Makefile to point to location of GCC on your system.

### Notes:
  1. The code was tested with OpenCV 3.X and GStreamer on Fedora 23 with version of gcc 5.3.0.

## Setup Walkthrough (01/2017)
### by Aza Tulepbergenov
Below I will enumerate the steps I took to install dependencies on clean version of Fedora 23.
  1. OpenCV
     * Install via [this](https://github.com/jayrambhia/Install-OpenCV) script.
  2. GStreamer
     * GStreamer should come pre-packaged with your OS. However, you may be missing some required libraries (i.e. libav).
     * To check, open any video file and confirm that you can play it. If you can, then likely, your system has everything.
     * If not, then you will need to install it. I recommend using package manager to do it. I did: ```sudo dnf install *-libav```
  3. Install appropriate GCC version. At the time of Jan. 2017, I was using CUDA 8, which relied on versions of GCC greater less than or equal to 5. My machine had GCC 6. I built and installed GCC 5.3.0 from source (it was quite an interesting task!)
      * Download [GCC source files](https://gcc.gnu.org/mirrors.html)
      * Go through the **Setup, Configure, Build, Install** steps outlined in this [white paper](http://www.tellurian.com.au/whitepapers/multiplegcc.php)
      * I also found [this video](https://www.youtube.com/watch?v=aR5JJWzqVx8) helpful.
  4. Fix the ```CXX``` variable in the Makefile to point to location of GCC on your system.
  5. ```make```

## Usage
In order to verify the implementation of our VIBE against original VIBE follow the steps below.

1. Obtain an executable for our VIBE by typing
        make VIBE-NaiveSerial-CPP

2. Also obtain executables of original VIBE by
        cd vibe-sources
        make

3. In the main directory run the script

  ```perl run_VIBE.pl <name_of_our_VIBE_results> <name_of_original_VIBE_1> <name_of  _original_VIBE_2>```

4. After this make sure you have a blank black image of the size of the frame named pix  el_count.png and also the executable for vibe_genarate_map.cpp
  make vibe_generate_map.

5. Then run the script
  ```perl generate_map.pl 1 <name_of_our_VIBE_results> <name_of_original_VIBE_1> <name_of  _original_VIBE_2>```
  - to obtain foregroung counter map for our VIBE.

  ```perl generate_map.pl 2 <name_of_our_VIBE_results>``` <name_of_original_VIBE_1> <name_of  _original_VIBE_2>
  - to obtain first foreground counter map for original VIBE.

  ```perl generate_map.pl any number <name_of_our_VIBE_results> <name_of_original_VIBE_1> <name_of  _original_VIBE_2>```
  - to obtain second foreground counter map for original VIBE.
  In each case make sure the blank pixel_count.png image is present before you r  generate the maps.

6. You will obtain he foreground counter maps named
  * pixel_count_1.png - first foreground counter map for original VIBE
  * pixel_count_2.png - second foreground counter map for original VIBE
  * pixel_count_3.png - foregorund counter map for our VIBE implementation

7. Obtain executable for vibe_generate_histogram.cpp
  make vibe_genearte_histogram.cpp

8. Run it using

   ```./vibe_generate_histogram pixel_count_1.png pixel_count_2.png```

9. Depending on which two counter maps you have chosen to be read
  you will get corresponding list of counts for the histogram which can be
  entered into excel to obtain histogram graph.
