#include "Astar.h"
#include "DFS_BFS.h"
#include <iostream>
using namespace std;
#pragma warning(disable:4996)

//#define Astar
//# define Search
int main()
{

//#ifndef Astar
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
	path.push_back(start);
	bool return_flag=false;
	dfs(graph, start, end, return_flag,path);
	dfs1(graph, start, end);
	bfs(graph, start, end);
//#else

	cout << "\n\n";
	if (freopen("in_AStar.txt", "r",stdin)==NULL)
	{
		cerr << "Fail";
		exit(1);
	}
	Graph graph_AStar;
	createGraph(graph_AStar);
//	int start, end;
	
	cout << "Please choose initial node number:" << endl;
	cin >> start;
	cout << "Your input is " << start << endl;
	cout << "Please choose goal node number:" << endl;
	cin >> end;
	cout << "Your input is " << end << endl;
	vector<int> result_path;
	GraphNode init, final;
	init.nodeNum = start;
	final.nodeNum = end;
	
	AStarShortestPathSearch(graph_AStar,init ,final, result_path);
	displaySearchPath(result_path,graph_AStar);
//#endif
//	system("pause");
	cout << "Success!"<<endl;
	while (1);
	return 0;
}