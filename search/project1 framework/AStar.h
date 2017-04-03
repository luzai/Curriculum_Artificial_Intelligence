#ifndef __ASTARSEARCH__
#define __ASTARSEARCH__

#include<vector>
#include<queue>
#include <iostream>

#define N 50
#define MAX 1000
using namespace std;



/* Initialization of a grapgh for BFS/DFS search.
   the following information is required:
   1) number of nodes and non-zero edges in the graph;
   2) value of non-zero edge and corresponding node number;
   3) start node number and final node number for searching;
   (you may need to transform node symbol to digit number for convenience; e.g. 0 reprensents node A,1 represents node B, etc...)
*/


/* definition of node in the graph
*/
struct GraphNode{ 
	int nodeNum; //node number
	int g, h;  //g(n) and h(n) of the node, f=g+h for comparing priority.
	GraphNode* preGraphNode;  //pointer to the parent node of this node, useful for reconstructing search path. 

	bool operator==(GraphNode a){
		if (a.nodeNum == nodeNum)
			return true;
		else
			return false;
	}

	GraphNode(){  //initialization
		g = 0;
		h = 0;
		nodeNum = 0;
		preGraphNode = NULL;
	}
	GraphNode(int nodeNum) :nodeNum(nodeNum){
		g = 0;
		h = 0; preGraphNode = NULL;
	}
	GraphNode(int nodeNum, int h) :nodeNum(nodeNum), h(h){
		g = 0;
		preGraphNode = NULL;
	}
};
/* used for comparing priority of nodes in the priority queue
*/
struct cmpLarge{
	bool operator()(GraphNode a, GraphNode  b){
		return ((a.g + a.h) >= (b.g + b.h));
	}
};

struct Graph{
	int edges[N][N];  //adjacency matrix storing edge information
	int n;    //number of nodes in the graph, no more than maxn. 
	int e;    //number of edges.
	int H[N]; //heuristic function value from all nodes to the given goal node
	vector<GraphNode> graph_nodes_;
};
void createGraph(Graph &G);

/* Your A* search implementation for finding the shortest path in a graph:
NOTICE:
   The main A* search function is AStarShortestPathSearch( ), in which you need to implement A* search from given start node to goal node;
   Prioroty queue is used for constructing openlist, while vector is used for constructing closelist;

     graph: the input graph to be searched;
	 initNode: initial search node;
	 finalNode: goal search node;
	 resultPath: a vector which stores the node numbers in order along the shortest path found;

*/
void AStarShortestPathSearch( Graph& graph, GraphNode& initNode, GraphNode& finalNode, vector<int> &resultPath);

bool checkGoalNode(const GraphNode* resultState, const GraphNode* goalNode);

bool displaySearchPath(const vector<int> &resultPath, const Graph& MGraph);




#endif