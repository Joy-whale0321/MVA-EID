/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides examples for the training and testing of the
/// TMVA classifiers.
///
/// As input data is used a toy-MC sample consisting of four Gaussian-distributed
/// and linearly correlated input variables.
/// The methods to be used can be switched on and off by means of booleans, or
/// via the prompt command, for example:
///
///     root -l ./TMVAClassification.C\(\"Fisher,Likelihood\"\)
///
/// (note that the backslashes are mandatory)
/// If no method given, a default set of classifiers is used.
/// The output file "TMVA.root" can be analysed with the use of dedicated
/// macros (simply say: root -l <macro.C>), which can be conveniently
/// invoked through a GUI that will appear at the end of the run of this macro.
/// Launch the GUI via the command:
///
///     root -l ./TMVAGui.C
///
/// You can also compile and run the example with the following commands
///
///     make
///     ./TMVAClassification <Methods>
///
/// where: `<Methods> = "method1 method2"` are the TMVA classifier names
/// example:
///
///     ./TMVAClassification Fisher LikelihoodPCA BDT
///
/// If no method given, a default set is of classifiers is used
///
/// - Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Root Macro: TMVAClassification
///
/// \macro_output
/// \macro_code
/// \author Andreas Hoecker


#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int TMVAClassification( TString myMethodList = "" )
{
   // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the
   // corresponding lines from .rootrc

   // Methods to be processed can be given as an argument; use format:
   //
   //     mylinux~> root -l TMVAClassification.C\(\"myMethod1,myMethod2,myMethod3\"\)

   //---------------------------------------------------------------
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Cut optimisation
   Use["Cuts"]            = 0;
   Use["CutsD"]           = 0;
   Use["CutsPCA"]         = 0;
   Use["CutsGA"]          = 0;
   Use["CutsSA"]          = 0;
   //
   // 1-dimensional likelihood ("naive Bayes estimator")
   Use["Likelihood"]      = 0;
   Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
   Use["LikelihoodPCA"]   = 0; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
   Use["LikelihoodKDE"]   = 0;
   Use["LikelihoodMIX"]   = 0;
   //
   // Mutidimensional likelihood and Nearest-Neighbour methods
   Use["PDERS"]           = 0;
   Use["PDERSD"]          = 0;
   Use["PDERSPCA"]        = 0;
   Use["PDEFoam"]         = 0;
   Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
   Use["KNN"]             = 0; // k-nearest neighbour method
   //
   // Boosted Decision Trees
   Use["BDT"]             = 1; // uses Adaptive Boost
   Use["BDTG"]            = 0; // uses Gradient Boost
   Use["BDTB"]            = 0; // uses Bagging
   Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
   Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting
   //
   // Linear Discriminant Analysis
   Use["LD"]              = 1; // Linear Discriminant identical to Fisher
   Use["Fisher"]          = 0;
   Use["FisherG"]         = 0;
   Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
   Use["HMatrix"]         = 0;
   //
   // Function Discriminant analysis
   Use["FDA_GA"]          = 0; // minimisation of user-defined function using Genetics Algorithm
   Use["FDA_SA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 0;
   Use["FDA_GAMT"]        = 0;
   Use["FDA_MCMT"]        = 0;
   //
   // Neural Networks (all are feed-forward Multilayer Perceptrons)
   Use["MLP"]             = 1; // Recommended ANN
   Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
   Use["MLPBNN"]          = 0; // Recommended ANN with BFGS training method and bayesian regulator
   Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
   Use["TMlpANN"]         = 0; // ROOT's own ANN
#ifdef R__HAS_TMVAGPU
   Use["DNN_GPU"]         = 0; // CUDA-accelerated DNN training.
#else
   Use["DNN_GPU"]         = 0;
#endif

#ifdef R__HAS_TMVACPU
   Use["DNN_CPU"]         = 1; // Multi-core accelerated DNN.
#else
   Use["DNN_CPU"]         = 1;
#endif
   //
   // Support Vector Machine
   Use["SVM"]             = 1;
   //
   // Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
   Use["RuleFit"]         = 0;
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassification" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return 1;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // Here the preparation phase begins

   // Read training and test data
   // (it is also possible to use ASCII format as input -> see TMVA Users Guide)
   TFile *input(0);
  // TString fname = "./tmva_class_example.root";
 //  TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_12.root";// the first 3 vars are used
  // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_6.root";// the first 3 vars are used
  // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt6_12.root";// the first 3 vars are used
   // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_5.root";// the first 3 vars are used
    // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt5_8.root";// the first 3 vars are used
   //  TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt8_12.root";// the first 3 vars are used

   // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_4.root";// the first 3 vars are used
  // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt4_6.root";// the first 3 vars are used
   // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt6_8.root";// the first 3 vars are used
   //  TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt8_10.root";// the first 3 vars are used
  //   TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt10_12.root";// the first 3 vars are used

     // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_12_embed.root";// the first 3 vars are used
     //  TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_5_embed.root";// the first 3 vars are used
      //  TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt5_8_embed.root";// the first 3 vars are used
       // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt8_12_embed.root";// the first 3 vars are used
       // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt2_6_embed.root";// the first 3 vars are used
         TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data2/MVAdata_7vars_e3x3_cutpt6_12_embed.root";// the first 3 vars are used
  
  
  
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
      std::cout << "Open: local MVAdata" << std::endl;//weihu
   }
   else {
      TFile::SetCacheFileDir(".");
      input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
      std::cout << "Open: cern online MVAdata" << std::endl;//weihu
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;

   // Register the training and test trees
   
  // TTree *signalTree     = (TTree*)input->Get("TreeS");
  // TTree *background     = (TTree*)input->Get("TreeB");

  // TTree *signalTree     = (TTree*)input->Get("TreeSall");//weihu

  TTree *signalTree     = (TTree*)input->Get("TreeSelectron");//weihu
   //TTree *signalTree     = (TTree*)input->Get("TreeSpositron");//weihu

    TTree *background     = (TTree*)input->Get("TreeBallN");//weihu
  // TTree *background     = (TTree*)input->Get("TreeBallP");//weihu

  // TTree *background     = (TTree*)input->Get("TreeBpion");//weihu

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   //TString outfileName( "TMVA.root" );
   // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN.root" );//weihu

   // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_12.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_6.root" );//weihu
   //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt6_12.root" );//weihu
  // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_5.root" );//weihu
   //TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt5_8.root" );//weihu
   //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt8_12.root" );//weihu

   //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_12.root" );//weihu
   //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_6.root" );//weihu
   // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt6_12.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_5.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt5_8.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt8_12.root" );//weihu

     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_12_4vars.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_12_4vars.root" );//weihu

      // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_12_pion.root" );//weihu

    //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_4.root" );//weihu
    //TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt4_6.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt6_8.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt8_10.root" );//weihu
    // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt10_12.root" );//weihu

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////embed
    //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_12_embed.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_5_embed.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt5_8_embed.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt8_12_embed.root" );//weihu
     //TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt2_6_embed.root" );//weihu
       TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allN_cutpt6_12_embed.root" );//weihu
  
    //  TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_12_embed.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_5_embed.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt5_8_embed.root" );//weihu
     // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt8_12_embed.root" );//weihu
      // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt2_6_embed.root" );//weihu
      // TString outfileName( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/training_output/TMVA_allP_cutpt6_12_embed.root" );//weihu
     
  
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory is
   // the only TMVA object you have to interact with
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_12");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_6");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt6_12");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_5");//weihu
   //TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt5_8");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt8_12");//weihu

  //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_12");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_6");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt6_12");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_5");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt5_8");//weihu
 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt8_12");//weihu

 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_12_4vars");//weihu
 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_12_4vars");//weihu

 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_12_pion");//weihu

 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_4");//weihu
 // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt4_6");//weihu
 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt6_8");//weihu
 //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt8_10");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt10_12");//weihu

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////embed
    // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_12_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_5_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt5_8_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt8_12_embed");//weihu
  // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt2_6_embed");//weihu
    TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allN_cutpt6_12_embed");//weihu

    //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_12_embed");//weihu
    //  TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_5_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt5_8_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt8_12_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt2_6_embed");//weihu
   // TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_allP_cutpt6_12_embed");//weihu
  
   
   // If you wish to modify default settings
   // (please check "src/Config.h" to see all available global options)
   //
   //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   
   //dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
  // dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
   dataloader->AddVariable( "var1", "Variable 1", "units", 'F' );//weihu
   dataloader->AddVariable( "var2", "Variable 2", "units", 'F' );//weihu
   dataloader->AddVariable( "var3", "Variable 3", "units", 'F' );//weihu
  // dataloader->AddVariable( "var4", "Variable 4", "units", 'F' );//weihu
  // dataloader->AddVariable( "var5", "Variable 5", "units", 'F' );//weihu
  // dataloader->AddVariable( "var6", "Variable 6", "units", 'F' );//weihu

   // You can add so-called "Spectator variables", which are not used in the MVA training,
   // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
   // input variables, the response values of all trained MVAs, and the spectator variables

   dataloader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
   dataloader->AddSpectator( "spec2 := var1*3",  "Spectator 2", "units", 'F' );


   // global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

   // You can add an arbitrary number of signal or background trees
   dataloader->AddSignalTree    ( signalTree,     signalWeight );
   dataloader->AddBackgroundTree( background, backgroundWeight );

   // To give different trees for training and testing, do as follows:
   //
   //     dataloader->AddSignalTree( signalTrainingTree, signalTrainWeight, "Training" );
   //     dataloader->AddSignalTree( signalTestTree,     signalTestWeight,  "Test" );

   // Use the following code instead of the above two or four lines to add signal and background
   // training and test events "by hand"
   // NOTE that in this case one should not give expressions (such as "var1+var2") in the input
   //      variable definition, but simply compute the expression before adding the event
   // ```cpp
   // // --- begin ----------------------------------------------------------
   // std::vector<Double_t> vars( 4 ); // vector has size of number of input variables
   // Float_t  treevars[4], weight;
   //
   // // Signal
   // for (UInt_t ivar=0; ivar<4; ivar++) signalTree->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
   // for (UInt_t i=0; i<signalTree->GetEntries(); i++) {
   //    signalTree->GetEntry(i);
   //    for (UInt_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
   //    // add training and test events; here: first half is training, second is testing
   //    // note that the weight can also be event-wise
   //    if (i < signalTree->GetEntries()/2.0) dataloader->AddSignalTrainingEvent( vars, signalWeight );
   //    else                              dataloader->AddSignalTestEvent    ( vars, signalWeight );
   // }
   //
   // // Background (has event weights)
   // background->SetBranchAddress( "weight", &weight );
   // for (UInt_t ivar=0; ivar<4; ivar++) background->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
   // for (UInt_t i=0; i<background->GetEntries(); i++) {
   //    background->GetEntry(i);
   //    for (UInt_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
   //    // add training and test events; here: first half is training, second is testing
   //    // note that the weight can also be event-wise
   //    if (i < background->GetEntries()/2) dataloader->AddBackgroundTrainingEvent( vars, backgroundWeight*weight );
   //    else                                dataloader->AddBackgroundTestEvent    ( vars, backgroundWeight*weight );
   // }
   // // --- end ------------------------------------------------------------
   // ```
   // End of tree registration

   // Set individual event weights (the variables must exist in the original TTree)
   // -  for signal    : `dataloader->SetSignalWeightExpression    ("weight1*weight2");`
   // -  for background: `dataloader->SetBackgroundWeightExpression("weight1*weight2");`
   //dataloader->SetBackgroundWeightExpression( "weight" );
  // dataloader->SetBackgroundWeightExpression( "weightantiproton" );//weihu
  // dataloader->SetBackgroundWeightExpression( "weightpion" );//weihu
  // dataloader->SetBackgroundWeightExpression( "weightKion" );//weihu
  
   dataloader->SetBackgroundWeightExpression( "weightBallN" );//weihu
   // dataloader->SetBackgroundWeightExpression( "weightBallP" );//weihu

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = "var1>0.0 && var1<10.0 && var2>0.0 && var2<10.0 && var3>0.0 && var3<20.0 "; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = "var1>0.0 && var1<10.0 && var2>0.0 && var2<10.0 && var3>0.0 && var3<20.0 "; // for example: TCut mycutb = "abs(var1)<0.5";

  // TCut mycuts = "var1>0.0 && var1<10.0 && var2>0.0 && var2<10.0 && var3>0.0 && var3<20.0  && var6>2.0 && var3<12.0"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
  // TCut mycutb = "var1>0.0 && var1<10.0 && var2>0.0 && var2<10.0 && var3>0.0 && var3<20.0  && var6>2.0 && var3<12.0"; // for example: TCut mycutb = "abs(var1)<0.5";

   // Tell the dataloader how to use the training and testing events
   //
   // If no numbers of events are given, half of the events in the tree are used
   // for training, and the other half for testing:
   //
   //    dataloader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
   //
   // To also specify the number of testing events, use:
   //
   //    dataloader->PrepareTrainingAndTestTree( mycut,"NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
  // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=3000:nTrain_Background=5500:nTest_Signal=3000:nTest_Background=5500:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP Smax=6000; Bmax=12000
 //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2600:nTrain_Background=5500:nTest_Signal=2600:nTest_Background=5500:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN Smax=6000; Bmax=12000
 
 //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=13000:nTrain_Background=31000:nTest_Signal=13000:nTest_Background=31000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt2_12 Smax=27201; Bmax=62892
 //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=6000:nTrain_Background=13000:nTest_Signal=6000:nTest_Background=13000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt2_6 Smax=12217; Bmax=27011
 //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=7000:nTrain_Background=17000:nTest_Signal=7000:nTest_Background=17000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt6_12 Smax=14984; Bmax=35881
  // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4500:nTrain_Background=10000:nTest_Signal=4500:nTest_Background=10000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt2_5 Smax=9290; Bmax=20377
 //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4000:nTrain_Background=9000:nTest_Signal=4000:nTest_Background=9000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt5_8 Smax=8451; Bmax=19225
  //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4700:nTrain_Background=11000:nTest_Signal=4700:nTest_Background=11000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt8_12 Smax=9460; Bmax=23290
  
  //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=12400:nTrain_Background=30300:nTest_Signal=12400:nTest_Background=30300:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_12 Smax=24984; Bmax=60673; 4 vars
  //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=12200:nTrain_Background=30000:nTest_Signal=12200:nTest_Background=30000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_12 Smax=24984; Bmax=60673
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=5000:nTrain_Background=13000:nTest_Signal=5000:nTest_Background=13000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_6 Smax=11289; Bmax=26135
  //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=6200:nTrain_Background=17000:nTest_Signal=6200:nTest_Background=17000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt6_12 Smax=13695; Bmax=34538
 //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4300:nTrain_Background=9500:nTest_Signal=4300:nTest_Background=9500:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_5 Smax=8602; Bmax=19846
  // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=3800:nTrain_Background=9000:nTest_Signal=3800:nTest_Background=9000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt5_8 Smax=7704; Bmax=18270
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4200:nTrain_Background=11000:nTest_Signal=4200:nTest_Background=11000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt8_12 Smax=8678; Bmax=22557
   //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=12200:nTrain_Background=10000:nTest_Signal=12200:nTest_Background=10000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  pion cutpt2_12 Smax=24984; Bmax=21312
 
  //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2900:nTrain_Background=6500:nTest_Signal=2900:nTest_Background=6500:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_4 Smax=5869; Bmax=13311
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2700:nTrain_Background=6400:nTest_Signal=2700:nTest_Background=6400:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt4_6 Smax=5421; Bmax=12824
  // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2500:nTrain_Background=5900:nTest_Signal=2500:nTest_Background=5900:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt6_8 Smax=5017; Bmax=11981
 //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2300:nTrain_Background=5800:nTest_Signal=2300:nTest_Background=5800:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt8_10 Smax=4742; Bmax=11601
 //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=1950:nTrain_Background=5400:nTest_Signal=1950:nTest_Background=5400:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt10_12 Smax=3936; Bmax=10956

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////embed
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=7000:nTrain_Background=14000:nTest_Signal=7000:nTest_Background=14000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_12_embed Smax=16171; Bmax=30785
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2800:nTrain_Background=6400:nTest_Signal=2800:nTest_Background=6400:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_5_embed Smax=5694; Bmax=13129
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2550:nTrain_Background=3600:nTest_Signal=2550:nTest_Background=3600:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt5_8_embed Smax=5205; Bmax=8725
   // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2600:nTrain_Background=4000:nTest_Signal=2600:nTest_Background=4000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt8_12_embed Smax=5272; Bmax=8904
 // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=3500:nTrain_Background=7500:nTest_Signal=3500:nTest_Background=7500:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt2_6_embed Smax=7533; Bmax=16009
  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4300:nTrain_Background=7000:nTest_Signal=4300:nTest_Background=7000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allN cutpt6_12_embed Smax=8638; Bmax=14749



  //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=4500:nTrain_Background=14000:nTest_Signal=4500:nTest_Background=14000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt2_12_embed Smax=9646; Bmax=32494
   // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=1800:nTrain_Background=6400:nTest_Signal=1800:nTest_Background=6400:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt2_5_embed Smax=3634; Bmax=13374
  //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=1500:nTrain_Background=4000:nTest_Signal=1500:nTest_Background=4000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt5_8_embed Smax=3101; Bmax=9616
  //  dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=1450:nTrain_Background=4000:nTest_Signal=1450:nTest_Background=4000:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt8_12_embed Smax=2911; Bmax=9504
  // dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2100:nTrain_Background=7500:nTest_Signal=2100:nTest_Background=7500:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt2_6_embed Smax=4677; Bmax=16648
  //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=2400:nTrain_Background=7700:nTest_Signal=2400:nTest_Background=7700:SplitMode=Random:NormMode=NumEvents:!V" );//weihu  allP cutpt6_12_embed Smax=4969; Bmax=15846

 
 
   // ### Book MVA methods
   //
   // Please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
   // it is possible to preset ranges in the option string in which the cut optimisation should be done:
   // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

   // Cut optimisation
   if (Use["Cuts"])
      factory->BookMethod( dataloader, TMVA::Types::kCuts, "Cuts",
                           "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );

   if (Use["CutsD"])
      factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsD",
                           "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=Decorrelate" );

   if (Use["CutsPCA"])
      factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsPCA",
                           "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=PCA" );

   if (Use["CutsGA"])
      factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsGA",
                           "H:!V:FitMethod=GA:CutRangeMin[0]=-10:CutRangeMax[0]=10:VarProp[1]=FMax:EffSel:Steps=30:Cycles=3:PopSize=400:SC_steps=10:SC_rate=5:SC_factor=0.95" );

   if (Use["CutsSA"])
      factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsSA",
                           "!H:!V:FitMethod=SA:EffSel:MaxCalls=150000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

   // Likelihood ("naive Bayes estimator")
   if (Use["Likelihood"])
      factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "Likelihood",
                           "H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );

   // Decorrelated likelihood
   if (Use["LikelihoodD"])
      factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodD",
                           "!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" );

   // PCA-transformed likelihood
   if (Use["LikelihoodPCA"])
      factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodPCA",
                           "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" );

   // Use a kernel density estimator to approximate the PDFs
   if (Use["LikelihoodKDE"])
      factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodKDE",
                           "!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" );

   // Use a variable-dependent mix of splines and kernel density estimator
   if (Use["LikelihoodMIX"])
      factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodMIX",
                           "!H:!V:!TransformOutput:PDFInterpolSig[0]=KDE:PDFInterpolBkg[0]=KDE:PDFInterpolSig[1]=KDE:PDFInterpolBkg[1]=KDE:PDFInterpolSig[2]=Spline2:PDFInterpolBkg[2]=Spline2:PDFInterpolSig[3]=Spline2:PDFInterpolBkg[3]=Spline2:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" );

   // Test the multi-dimensional probability density estimator
   // here are the options strings for the MinMax and RMS methods, respectively:
   //
   //      "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.2:KernelEstimator=Gauss:GaussSigma=0.3" );
   //      "!H:!V:VolumeRangeMode=RMS:DeltaFrac=3:KernelEstimator=Gauss:GaussSigma=0.3" );
   if (Use["PDERS"])
      factory->BookMethod( dataloader, TMVA::Types::kPDERS, "PDERS",
                           "!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );

   if (Use["PDERSD"])
      factory->BookMethod( dataloader, TMVA::Types::kPDERS, "PDERSD",
                           "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=Decorrelate" );

   if (Use["PDERSPCA"])
      factory->BookMethod( dataloader, TMVA::Types::kPDERS, "PDERSPCA",
                           "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA" );

   // Multi-dimensional likelihood estimator using self-adapting phase-space binning
   if (Use["PDEFoam"])
      factory->BookMethod( dataloader, TMVA::Types::kPDEFoam, "PDEFoam",
                           "!H:!V:SigBgSeparate=F:TailCut=0.001:VolFrac=0.0666:nActiveCells=500:nSampl=2000:nBin=5:Nmin=100:Kernel=None:Compress=T" );

   if (Use["PDEFoamBoost"])
      factory->BookMethod( dataloader, TMVA::Types::kPDEFoam, "PDEFoamBoost",
                           "!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T" );

   // K-Nearest Neighbour classifier (KNN)
   if (Use["KNN"])
      factory->BookMethod( dataloader, TMVA::Types::kKNN, "KNN",
                           "H:nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" );

   // H-Matrix (chi2-squared) method
   if (Use["HMatrix"])
      factory->BookMethod( dataloader, TMVA::Types::kHMatrix, "HMatrix", "!H:!V:VarTransform=None" );

   // Linear discriminant (same as Fisher discriminant)
   if (Use["LD"])
      factory->BookMethod( dataloader, TMVA::Types::kLD, "LD", "H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

   // Fisher discriminant (same as LD)
   if (Use["Fisher"])
      factory->BookMethod( dataloader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

   // Fisher with Gauss-transformed input variables
   if (Use["FisherG"])
      factory->BookMethod( dataloader, TMVA::Types::kFisher, "FisherG", "H:!V:VarTransform=Gauss" );

   // Composite classifier: ensemble (tree) of boosted Fisher classifiers
   if (Use["BoostedFisher"])
      factory->BookMethod( dataloader, TMVA::Types::kFisher, "BoostedFisher",
                           "H:!V:Boost_Num=20:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=0.2:!Boost_DetailedMonitoring" );


   // Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit (or GA or SA)
   if (Use["FDA_MC"])
      factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MC",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );
  // if (Use["FDA_MC"])
  //    factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MC",
  //                         "H:!V:Formula=(0)+(1)*x0+(2)*x1:ParRanges=(-1,1);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );//weihu
      

   if (Use["FDA_GA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options])
      factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_GA",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=100:Cycles=2:Steps=5:Trim=True:SaveBestGen=1" );

   if (Use["FDA_SA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options])
      factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_SA",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=SA:MaxCalls=15000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

   if (Use["FDA_MT"])
      factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MT",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

   if (Use["FDA_GAMT"])
      factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_GAMT",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:Cycles=1:PopSize=5:Steps=5:Trim" );

   if (Use["FDA_MCMT"])
      factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MCMT",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:SampleSize=20" );
                      
                           
   // TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
   if (Use["MLP"])
      factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );

   if (Use["MLPBFGS"])
      factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBFGS", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:!UseRegulator" );

   if (Use["MLPBNN"])
      factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=60:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" ); // BFGS training with bayesian regulators


   // Multi-architecture DNN implementation.
   if (Use["DNN_CPU"] or Use["DNN_GPU"]) {
      // General layout.
      TString layoutString ("Layout=TANH|128,TANH|128,TANH|128,LINEAR");

      // Define Training strategy. One could define multiple stratgey string separated by the "|" delimiter 

      TString trainingStrategyString = ("TrainingStrategy=LearningRate=1e-2,Momentum=0.9,"
                                        "ConvergenceSteps=20,BatchSize=100,TestRepetitions=1,"
                                        "WeightDecay=1e-4,Regularization=None,"
                                        "DropConfig=0.0+0.5+0.5+0.5");

      // General Options.
      TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
                          "WeightInitialization=XAVIERUNIFORM");
      dnnOptions.Append (":"); dnnOptions.Append (layoutString);
      dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

      // Cuda implementation.
      if (Use["DNN_GPU"]) {
         TString gpuOptions = dnnOptions + ":Architecture=GPU";
         factory->BookMethod(dataloader, TMVA::Types::kDL, "DNN_GPU", gpuOptions);
      }
      // Multi-core CPU implementation.
      if (Use["DNN_CPU"]) {
         TString cpuOptions = dnnOptions + ":Architecture=CPU";
         factory->BookMethod(dataloader, TMVA::Types::kDL, "DNN_CPU", cpuOptions);
      }
   }

   // CF(Clermont-Ferrand)ANN
   if (Use["CFMlpANN"])
      factory->BookMethod( dataloader, TMVA::Types::kCFMlpANN, "CFMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...

   // Tmlp(Root)ANN
   if (Use["TMlpANN"])
      factory->BookMethod( dataloader, TMVA::Types::kTMlpANN, "TMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N:LearningMethod=BFGS:ValidationFraction=0.3"  ); // n_cycles:#nodes:#nodes:...

   // Support Vector Machine
   if (Use["SVM"])
      factory->BookMethod( dataloader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" );

   // Boosted Decision Trees
   if (Use["BDTG"]) // Gradient Boost
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG",
                           "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );

   if (Use["BDT"])  // Adaptive Boost
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDT",
                           "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

   if (Use["BDTB"]) // Bagging
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTB",
                           "!H:!V:NTrees=400:BoostType=Bagging:SeparationType=GiniIndex:nCuts=20" );

   if (Use["BDTD"]) // Decorrelation + Adaptive Boost
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTD",
                           "!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:VarTransform=Decorrelate" );

   if (Use["BDTF"])  // Allow Using Fisher discriminant in node splitting for (strong) linearly correlated variables
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTF",
                           "!H:!V:NTrees=50:MinNodeSize=2.5%:UseFisherCuts:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20" );

   // RuleFit -- TMVA implementation of Friedman's method
   if (Use["RuleFit"])
      factory->BookMethod( dataloader, TMVA::Types::kRuleFit, "RuleFit",
                           "H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );

   // For an example of the category classifier usage, see: TMVAClassificationCategory
   //
   // --------------------------------------------------------------------------------------------------
   //  Now you can optimize the setting (configuration) of the MVAs using the set of training events
   // STILL EXPERIMENTAL and only implemented for BDT's !
   //
   //     factory->OptimizeAllMethods("SigEffAtBkg0.01","Scan");
   //     factory->OptimizeAllMethods("ROCIntegral","FitGA");
   //
   // --------------------------------------------------------------------------------------------------

   // Now you can tell the factory to train, test, and evaluate the MVAs
   //
   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;
   delete dataloader;
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

   return 0;
}

int main( int argc, char** argv )
{
   // Select methods (don't look at this code - not of interest)
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   return TMVAClassification(methodList);
}
