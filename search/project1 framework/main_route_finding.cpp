#include "Astar.h"
#include "DFS_BFS.h"
#include <iostream>

#pragma warning(disable:4996)

//#define Astar
# define Search
int main()
{

#ifndef Astar
	if (freopen("in.txt", "r",stdin)==NULL)
	{
		cerr << "Fail";
		exit(1);
	}

	MGraph graph;

	createMGraph(graph);

	int start, end;

	cout << "Please choose initial node number:" << endl;
	cin >> start;
	cout << "Your input is " << start << endl;
	cout << "Please choose goal node number:" << endl;
	cin >> end;
	cout << "Your input is " << end << endl;

	bool endflag = false;
	vector<int> path;
	bool return_flag=false;
	dfs(graph, start, end, return_flag,path);
	dfs1(graph, start, end);
	bfs(graph, start, end);
#else
	if (freopen("in2.txt", "r",stdin)==NULL)
	{
		cerr << "Fail";
		exit(1);
	}
	Graph graph;
	createGraph(graph);
	int start, end;
	
	cout << "Please choose initial node number:" << endl;
	cin >> start;
	cout << "Your input is " << start << endl;
	cout << "Please choose goal node number:" << endl;
	cin >> end;
	cout << "Your input is " << end << endl;
	vector<int> result_path;
	for (auto it = graph.graph_nodes_.begin(); it != graph.graph_nodes_.end(); it++)
	{
		cout << it->nodeNum;
	}
	AStarShortestPathSearch(graph, graph.graph_nodes_[start],graph.graph_nodes_[end], result_path);
	displaySearchPath(result_path,graph);
#endif

	getchar();
	return 0;
}