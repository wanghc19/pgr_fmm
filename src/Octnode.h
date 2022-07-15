#ifndef OCTNODE
#define OCTNODE
#include "Cube.h"
#include <stdio.h>
#include <vector>
#include "Constants.h"

using namespace std;
class OctNode;

struct GridData{
	float value;
	//float smoothWidth;
	//float querySmoothWidth;
	float width;
	GridData* adjacent[Cube::NEIGHBORS];
	long long adjacentKey[Cube::NEIGHBORS];
	float coords[3];
	long long key;
	int maxDepth;
	int minDepth;
	int sampleCalculated;
	bool vectorSet;
	OctNode* node;
	float normalVar;
	GridData(){
		for (int i = 0; i < Cube::NEIGHBORS; i++){
			adjacent[i] = NULL;
			adjacentKey[i] = -1;
		}
		key = -1;
		value = 0;
		coords[0] = coords[1] = coords[2] = 0;
		maxDepth = -1;
		minDepth = DEPTH_LIMIT;
		//smoothWidth = 0;
		//querySmoothWidth = 0;
		sampleCalculated = 0;
		node = NULL;
		vectorSet = false;
		normalVar = 0;
	}
};


class NodeAdjacencyFunction{
public:
	virtual void Function(const OctNode* node1) = 0;
};


class OctNode{
private:
	int depth;
	int offset[3];
	
public:
	//long long gridCornerIdx[8];
	GridData* cornerGrid[Cube::CORNERS];
	int gridStartIdx;
	int gridEndIdx;
	vector<int> nodeIdx;
	//WeightedPoint weightedPoint;
	Point barycenter;
	bool hasSample;
	OctNode* parent;
	OctNode* children;
	int normalIdx;
	int mcIdx;
	float centerWeightContribution;
	float value;
private:
	void Index( const int& depth, const int offset[3], int& d, int off[3]);
	const OctNode* __faceNeighbor(const int& dir,const int& off) const;
	const OctNode* __edgeNeighbor(const int& o,const int i[2],const int idx[2]) const;
	OctNode* __faceNeighbor(const int& dir, const int& off, const int& forceChildren);

public:
	OctNode();
	void initChildren();
	long long getCornerIndex(const int& childNo, const int& maxDepth);
	long long getCornerIndex(const int& childNo, const int& maxDepth) const;
	int leaves();
	int nodes();
	int maxDepth();
	int Depth() const;
	OctNode* nextNode(OctNode* current = NULL);
	OctNode* nextBranch(OctNode* current);
	OctNode* nextLeaf(OctNode* currentLeaf = NULL);
	static int CompareForwardPointerDepths( const void* v1, const void* v2);
	const OctNode* faceNeighbor(const int& faceIndex) const;
	OctNode* faceNeighbor(const int& faceIndex, const int& forceChildren = 0);
	const OctNode* edgeNeighbor(const int& edgeIndex) const;
	void depthAndOffset( int& depth, int off[3]) const;
	void processNodeFaces(OctNode* node, NodeAdjacencyFunction* F, const int& fIndex, const int& processCurrent = 1);
	void __processNodeFaces(OctNode* node, NodeAdjacencyFunction* F, const int& cIndex1, const int& cIndex2, const int& cIndex3, const int& cIndex4);
	void centerAndWidth(float* center, float& width);
};

class SortedNodes{
public:
	OctNode** treeNodes;
	int *nodeCount;
	int maxDepth;
	SortedNodes();
	~SortedNodes();
	void set(OctNode& root, const int& setIndex);
};

class RootInfo{
public:
	const OctNode* node;
	int edgeIndex;
	long long key;
};

class Neighbors{
public:
	OctNode* neighbors[3][3][3];
	Neighbors(void);
	void clear(void);
};

class NeighborKey{
public:
	Neighbors* neighbors;

	NeighborKey(void);
	~NeighborKey(void);

	void set(const int& depth);
	Neighbors& setNeighbors(OctNode* node);
	Neighbors& getNeighbors(OctNode* node);
	
};


#endif