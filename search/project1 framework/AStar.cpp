//#include "Astar.h"
//#include <iostream>
//#include <vector>
//#include <queue>
//
//using namespace std;
//
//char AStarNode[6] = { 'A', 'B', 'C', 'D', 'E', 'F' };
//
//
//void createGraph(Graph &G)
//{
//	cout << "enter the number of nodes in the graph:" << endl;
//	cin >> G.n;
//	cout << "your input is " << G.n << endl;
//	cout << "enter the number of edges in the graph:" << endl;
//
//	cin >> G.e;
//	cout << "your input is " << G.e << endl;
//	int i, j;
//	int s, t; //start and end node number
//	int v;  //value of edge between node s and t
//	for (i = 0; i<G.n; i++)  
//	{
//		G.H[i] = 0;
//		for (j = 0; j<G.n; j++)
//		{
//			G.edges[i][j] = 0;
//		}
//	}
//	cout << "enter the value of edge/weights and corresponding node number" << endl;
//	cout << "(e.g. 1 3 10 means the edge from node 1 to node 3 is 10) : " << endl;
//	cout << "first: " << endl;
//	for (i = 0; i<G.e; i++) //set value for non-zero edges
//	{
//		cin >> s >> t >> v;         
//		cout << "your input is " << " " << s << " " << t << " " << v << endl;
//		G.edges[s][t] = v;
//		cout << "next:" << endl;
//	}
//	for (i = 0; i < G.n; i++)
//	{
//		int h;
//		cin >> h;
//		G.H[i] = h;
//	}
//	for (i = 0; i < G.n; i++)
//	{
//		GraphNode graph_node_t(i,G.H[i]);
//		G.graph_nodes_.push_back(graph_node_t);
//	}
//	for (i = 0; i < G.n; i++)
//	{
//		for (j = 0; j <= G.n; j++)
//		{
//			if (G.edges[i][j] != 0)
//			{
//				
//			}
//		}
//	}
//}
//vector<GraphNode&> getSuccessors(GraphNode& current, Graph& graph)
//{
//	vector<GraphNode&> successors;
//	int curr_num = current.nodeNum;
//	for (int i = 0; i < graph.n; i++)
//	{
//		if (graph.edges[curr_num][i] != 0)
//		{
//			successors.push_back(graph.graph_nodes_[i]);
//		}
//	}
//	return successors;
//}
//bool checkInCloseList(GraphNode& node, vector<GraphNode&> close)
//{
//	for (auto it = close.begin(); it != close.end(); it++)
//	{
//		if (node.nodeNum != it->nodeNum)
//			return true;
//	}
//	return false;
//}
//
//bool checkInOpenList(GraphNode& node, priority_queue<GraphNode&, vector<GraphNode&>, cmpLarge> open)
//{
//	vector<GraphNode&> node_tmp;
//	bool flag=false;
//	while (!open.empty())
//	{	
//		GraphNode& tmp = open.top();
//		if (tmp.nodeNum == node.nodeNum)
//			flag = true;
//		node_tmp.push_back(tmp);
//		open.pop();
//	}
//	for (auto it = node_tmp.begin(); it != node_tmp.end();it++)
//	{
//		open.push(*it);
//	}
//	return flag;
//}
////
////void AStarShortestPathSearch( Graph& graph, 
////	GraphNode& initNode, 
////	GraphNode& finalNode, 
////	vector<int> &resultPath){
////
////	//push states to priority queue,and pop according to priority
////	priority_queue<GraphNode&, vector<GraphNode&>, cmpLarge> openlist; 
////	vector<GraphNode&> closelist;
////	openlist.push(initNode);
////	
////	while (!openlist.empty())
////	{
////		GraphNode& current = openlist.top();
////		openlist.pop();
////		if (current.nodeNum == finalNode.nodeNum)
////		{
////			GraphNode* curr = &(current);
////			while (curr!= nullptr)
////			{
////				resultPath.push_back(curr->nodeNum);
////				curr = curr->preGraphNode;
////			}
////			return;
////			// construct path in resultPath
////		}
////		closelist.push_back(current);
////		vector<GraphNode&> successors = getSuccessors(current, graph);
////		for (auto successor_it = successors.begin(); successor_it != successors.end(); successor_it++)
////		{
////			successor_it->preGrapghNode = &(current);
////			if( checkInCloseList(*successor_it, closelist))
////			{
////				continue;;
////			}
////			if (!checkInOpenList(*successor_it, openlist))
////			{
////				successor_it->g = current.g + graph.edges[current.nodeNum][successor_it->nodeNum];
////				openlist.push(*successor_it);
////			}
////			else if (successor_it->g >current.g + graph.edges[current.nodeNum][successor_it->nodeNum])
////			{
////				successor_it->g = current.g + graph.edges[current.nodeNum][successor_it->nodeNum];
////				successor_it->preGrapghNode = &(current);
////			}
////			
////		}
////	}
////	/*
////	openlist.push(&initNode);
////	while openlist is not empty:
////
////		current = openlist.top()
////		openlist.pop();
////
////		if (current==goal node)
////			return reconstruct_path(current, initNode, resultPath)
////
////		closelist.push_back(current)
////
////		successors=getSuccessors(current)
////			for (i = 0; i < successors.size(); ++i)
////
////				if successors[i] in closelist //if the successor has already been explored
////					continue;
////				if successors[i] not in openlist //check if the successor is not in the openlist
////					compute g, h and preGraphNode of successors[i]
////					push successors[i] to openlist
////				
////				else if g of successors[i] from current path is bigger than previous g of successors[i]    
////				//remember to set proper value to g and h 
////					continue;  
////
////				update g, h and preGraphNode of successors[i]
////	*/		
////}
//
//bool checkGoalNode(const GraphNode* resultState, const GraphNode* goalNode){
//	if (resultState->nodeNum == goalNode->nodeNum)
//		return true;
//
//	return false;
//}
//
///* display the search result
//   resultPath: a vector which stores the node numbers in order along the shortest path found
//   MGraph : the search graph, here it is used to compute the path length
//*/
////bool displaySearchPath(const vector<int> &resultPath, const Graph& MGraph){
////	int shortestPath = 0;
////	int num = int(resultPath.size());
////	if (resultPath.empty())
////		return false;
////	cout << "The shortest path is:" << endl;
////	for (int i = 0; i < num - 1; ++i){
////		cout << AStarNode[resultPath[i]] << "-> ";
////		shortestPath += MGraph.edges[resultPath[i]][resultPath[i + 1]];
////
////	}
////	cout << AStarNode[resultPath[num - 1]] << endl;
////
////	cout << "Path length: " << shortestPath << endl;
////
////	return true;
////}
//
//
//
