#!/usr/bin/perl
#
# This script will sweep through the parameters for our experiments
#
#

use strict;
#x_block:1024,y_block:1,x_factor:0,y_factor:0,x_tile:1280,y_tile:16,MemCpyCPUtoGPU:0.000000,KernelProcessing:0.000000,MemCpyGPUtoCPU:0.000000,Totaltime:0.000000

my $filename = $ARGV[0];

## a bit of configuration
my @headers = ('x_block','y_block','x_tile','y_tile','KernelProcessing');

open(my $datafile,'<',$filename)
  or die "Unable to open $filename : $!";

while(<$datafile>){
  my $line = $_;
  my @pair_data = split /,/,$line;
  my %local_data;
  foreach my $pair (@pair_data){
    my @data = split /:/,$pair;
    $local_data{$data[0]} = $data[1];
  }

  ## A single line has been read so now print it
  foreach my $header (@headers){
    print "$local_data{$header},";
  }
  print "\n";
}
