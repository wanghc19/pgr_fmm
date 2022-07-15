/*
Copyright (c) 2021, Authors of the anonymous ACMTOG submission "Surface Reconstruction from Point Clouds without Normals by Parametrizing the Gauss Formula"
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of Tsinghua University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "Octree.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <fstream>
#include "Geometry.h"

#include <CLI11.hpp>
#include <cnpy.h>

int main(int argc, char** argv) {

    std::string ply_suffix(".ply");
	std::string normalized_npy_suffix = "_normalized.npy";
	std::string query_npy_suffix("_for_query.npy");
    
	std::string inFileName;
	std::string outFileName;
	int minDepth = 1;
	int maxDepth = 10;
    
    CLI::App app("PGRExportQuery");
    app.add_option("-i", inFileName, "input filename of xyz format")->required();
	app.add_option("-o", outFileName, "output filename with no suffix")->required();
	app.add_option("-m", minDepth, "");
	app.add_option("-d", maxDepth, "");
	
    CLI11_PARSE(app, argc, argv);

	if (maxDepth < minDepth) {
		cout << "[In PGRExportQuery] WARNING: minDepth "
			 << minDepth
			 << " smaller than maxDepth "
			 << maxDepth
			 << ", ignoring given minDepth\n";
	}
		
	Octree tree;
	tree.setTree(inFileName, maxDepth, minDepth);//1382_seahorse2_p

	//*** Nodes for query are from gridDataVector *** START ***
	unsigned long N_grid_pts = tree.gridDataVector.size();
	unsigned long N_sample_pts = tree.samplePoints.size();
	std::vector<float> grid_coords;
	
	for(int grid_idx=0; grid_idx<N_grid_pts; grid_idx++) {
		grid_coords.push_back( tree.gridDataVector[grid_idx]->coords[0] );
		grid_coords.push_back( tree.gridDataVector[grid_idx]->coords[1] );
		grid_coords.push_back( tree.gridDataVector[grid_idx]->coords[2] );
	}

	// exporting normalized point samples as npy
	std::vector<float> pts_normalized;
	for (int i=0; i<N_sample_pts; i++){
		pts_normalized.push_back( tree.samplePoints[i].x );
		pts_normalized.push_back( tree.samplePoints[i].y );
		pts_normalized.push_back( tree.samplePoints[i].z );
	}

	cnpy::npy_save(outFileName + normalized_npy_suffix, &pts_normalized[0], {N_sample_pts, 3}, "w");
	std::cout << "[In PGRExportQuery] Normalizing the point cloud. Result saved to " << outFileName + normalized_npy_suffix <<std::endl;
	cnpy::npy_save(outFileName + query_npy_suffix, &grid_coords[0], {N_grid_pts, 3}, "w");
	std::cout << "[In PGRExportQuery] Exporting points on octree for query. Result saved to " << outFileName + query_npy_suffix <<std::endl;
}
