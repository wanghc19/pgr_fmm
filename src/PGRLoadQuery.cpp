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
#include <cstdlib>
#include <cstdio>
#include "Geometry.h"

#include <CLI11.hpp>

int main(int argc, char** argv) {

    std::string ply_suffix(".ply");
    std::string inFileName;
    std::string outFileName;
	std::string inGridValFileName;
	std::string inGridWidthFileName;
	float isovalue = 0.5;
	int minDepth = 1;
	int maxDepth = 10;
    
    CLI::App app("PGRLoadQuery");
    app.add_option("-i", inFileName, "input filename of xyz format")->required();
    app.add_option("-o", outFileName, "output filename of ply format")->required();
	app.add_option("--grid_val", inGridValFileName, "input filename of npy format for grid vals")->required();
	app.add_option("--grid_width", inGridWidthFileName, "input filename of npy format for grid widths")->required();
	app.add_option("-m", minDepth, "");
	app.add_option("-d", maxDepth, "");
	app.add_option("--isov", isovalue, "isovalue");

    CLI11_PARSE(app, argc, argv);

	if (maxDepth < minDepth) {
		cout << "[In PGRLoadQuery] WARNING: minDepth "
			 << minDepth
			 << " smaller than maxDepth "
			 << maxDepth
			 << ", ignoring given minDepth\n";
	}
	
	Octree tree;
	tree.setTree(inFileName, maxDepth, minDepth);//1382_seahorse2_p
	
	int N_grid = tree.gridDataVector.size();
	tree.loadImplicitFunctionFromNPY(inGridValFileName, N_grid);
	tree.loadGridWidthFromNPY(inGridWidthFileName, N_grid);
	
	std::cout << "[In PGRLoadQuery] Isovalue: " << isovalue << std::endl;

	CoredVectorMeshData mesh;
	tree.GetMCIsoTriangles(isovalue,  &mesh, 0, 1, /*add barycenter = */1, 0);
	char fileChar[255];
	strcpy(fileChar, (outFileName).c_str());
	tree.writePolygon2(&mesh, fileChar);
	std::cout << "[In PGRLoadQuery] Polygon Written to " << outFileName << std::endl;
}
