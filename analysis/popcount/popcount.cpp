#include <random>
#include <iostream>
#include <cmath>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "AdderTree.h"
#include "Adder.h"
#include "Definition.h"
using namespace std;
int main(){
  InputParameter inputParameter; Technology tech; MemCell cell;
  inputParameter.transistorType = conventional; inputParameter.deviceRoadmap = LSTP;
  inputParameter.temperature = param->temp; inputParameter.processNode = param->technode;
  tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);
  cout<<"node "<<param->technode<<" clk "<<param->clkFreq<<endl;
  for (int pr : {64,128,256}) {
    AdderTree at(inputParameter, tech, cell);
    at.Initialize(pr, 1, 1, param->clkFreq);
    at.CalculateArea(NULL, 128*param->featuresize*param->widthInFeatureSize1T1R, NONE);
    at.CalculateLatency(1, 0, 0);
    at.CalculatePower(1, 0);
    cout<<"PR "<<pr<<" area_um2 "<<at.area*1e12<<" E_per_op_pJ "<<at.readDynamicEnergy*1e12<<" lat_ns "<<at.readLatency*1e9<<" leak_uW "<<at.leakage*1e6<<endl;
  }
  return 0;
}
