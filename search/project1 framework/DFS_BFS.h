#ifndef __BFSDFSSEARCH__
#define __BFSDFSSEARCH__

#include<iostream>
#include <vector>

#define maxn 100

struct MGraph
{
	int edges[maxn][maxn];  //adjacency matrix storing edge information, if two nodes are not connected, corresponding edge is 0.
	int n;   //number of nodes in the graph, no more than maxn.   
	int e;    //number of edges.
};

/* Initialization of a grapgh for BFS/DFS search.
   the following information is required: 
   1) number of nodes and non-zero edges in the graph;
   2) value of non-zero edge and corresponding node number;
   3) start node number and final node number for searching; 
     (you may need to transform node symbol to digit number for convenience; e.g. 0 reprensents node A,1 represents node B, etc...)
*/
void createMGraph(MGraph &G);

//You may choose to implement recursive or non-recursive veision of DFS, or both if you want.

/* Recursive implementation of DFS for finding path from given start node to end node
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
   endflag: a bool variable for determining whether searching has reached the end node
*/ 
void dfs(MGraph G, int start, int end, bool& return_flag, std::vector<int> path);

/* Non-recursive implementation of DFS for finding path 
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
*/
void dfs1(MGraph G, int start, int end);

/* implementation of BFS for finding path 
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
*/
void bfs(MGraph G, int start, int end);

#endif