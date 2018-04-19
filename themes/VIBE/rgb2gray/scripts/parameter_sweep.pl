#!/usr/bin/perl

my @areas = (32,64,128,256,512,1024);
my @x_values = (32,16,8,4,2,1);

my $idx =$ARGV[0];

if($idx == 1)
{
 foreach my $a (@areas){
  foreach my $x (@x_values){
      my $y = $a/$x;
      my $output = `bin/rgb2gray-image-CUDA_int -f etc/CIMG0272.AVI -x $x -y $y -v 2>&1`;
      print $output;

   }
 }  
}

else{

  foreach my $a (@areas){
   foreach my $y (@y_values){
      my $x = $a/$y;
      my $output = `./VIBE-FusedParallel-CUDA -f sequence.ogv -x $x -y $y -g 1 -r 20 -o output.png 2>&1`;
      print $output

  }
 }
}


