#!/usr/bin/perl
#
# This script will sweep through the parameters for our experiments
#
#

use strict;

my @t_values = (1280,640,320,160,160);
my @q_values = (16,16,16,16,4);
my @areas = (32,64,128,256,512,1024);
my @y_values = (16,8,4,2,1);

my $idx;
for($idx=0;$idx<scalar(@t_values);$idx++){
  foreach my $a (@areas){
    foreach my $y (@y_values){
      my $x = $a/$y;
      my $output = `bin/rgb2gray-image-CUDA-1D-memorycoelacing_int -f ~vision/data/nobackup/NIH/CIMG0272.AVI -t $t_values[$idx] -q $q_values[$idx] -x $x -y $y -a 10 2>&1`;
      print $output;
      
    }
  }
}
