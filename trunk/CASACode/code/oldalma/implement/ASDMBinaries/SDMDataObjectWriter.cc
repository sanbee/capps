#include "SDMDataObjectWriter.h"
#include <algorithm>

#include "CommonDefines.h"

namespace asdmbinaries {

  const string SDMDataObjectWriter::MIMEBOUNDARY_1  = $MIMEBOUNDARY1;
  const string SDMDataObjectWriter::MIMEBOUNDARY_2  = $MIMEBOUNDARY2;

  const bool SDMDataObjectWriter::initClass_ = SDMDataObjectWriter::initClass();
  
  SDMDataObjectWriter::SDMDataObjectWriter(const string& uid, const string& title) {
    currentState_ = START;
    otype_ = STDOUT;
    uid_ = uid;
    title_ = title;
    preamble();
    done_ = false;
    sdmDataSubsetNum_ = 0;
  }
  
  SDMDataObjectWriter::SDMDataObjectWriter(ostringstream* oss, const string& uid,  const string& title) {
    currentState_ = START;
    otype_ = MEMORY;
    ofs_ = 0;
    oss_ = oss;
    uid_ = uid;
    title_ = title;
    preamble();
    done_ = false;
    sdmDataSubsetNum_ = 0;
  }
  
  SDMDataObjectWriter::SDMDataObjectWriter(ofstream* ofs, const string& uid,  const string& title) {
    currentState_ = START;
    otype_ = FILE;
    ofs_ = ofs;
    oss_ = 0;
    uid_ = uid;
    title_ = title;
    preamble();
    done_ = false;
    sdmDataSubsetNum_ = 0;
  }

  SDMDataObjectWriter::~SDMDataObjectWriter() {
    // Do nothing actually !
    // if (!done_) done();
  }

  void SDMDataObjectWriter::done() {
    checkState(T_DONE, "done");

    // Write MIME postamble.
    postamble();

    // And do some I/O management.    
    switch (otype_) {
    case STDOUT:
      // Do Nothing special
      break;
    case MEMORY:
      // Do Nothing special
      break;
    case FILE:
      // Do nothing special
      break;
    }
    done_ = true;
  }
  
  void SDMDataObjectWriter::output(const string& s) {
    switch (otype_) {
    case STDOUT:
      cout << s;
      break;

    case MEMORY:
      *oss_ << s;
      break;

    case FILE:
      *ofs_ << s;
      break;
    }
  }

  void SDMDataObjectWriter::outputln(const string& s) {
    output(s);
    outputln();
  }

  void SDMDataObjectWriter::output(const float* data, unsigned int numData) {
    switch (otype_) {

    case STDOUT:
      cout.write((const char*)data, numData*sizeof(float));
      break;

    case MEMORY:
      oss_->write((const char*)data, numData*sizeof(float));
      break;
      
    case FILE:
      ofs_->write((const char*)data, numData*sizeof(float));
      break;
    } 
  }

  void SDMDataObjectWriter::outputln(const float* data, unsigned int numData) {
    output(data, numData);
    outputln();
  } 

  void SDMDataObjectWriter::outputln() {
    output("\n");
  }

  void SDMDataObjectWriter::outputlnLocation(const string& name, const SDMDataSubset& sdmDataSubset) {
    outputln("Content-Location: " + sdmDataSubset.projectPath() +  name + ".bin");
  }

  void SDMDataObjectWriter::preamble() {
    outputln("MIME-Version: 1.0");
    outputln("Content-Type: multipart/mixed; boundary=\""+MIMEBOUNDARY_1+"\"; type=\"text/xml\"");
    outputln("Content-Description: " + title_ );
    outputln("Content-Location: uid:" + uid_.substr(4));
    outputln();
  }


  void SDMDataObjectWriter::postamble() {
    outputln("--"+MIMEBOUNDARY_1+"--");
  }


  void SDMDataObjectWriter::tpDataHeader(unsigned long long int startTime,
					 const string& execBlockUID,
					 unsigned int execBlockNum,
					 unsigned int scanNum,
					 unsigned int subscanNum,
					 unsigned int numberOfIntegrations,
					 unsigned int numAntenna,
					 SDMDataObject::DataStruct& dataStruct) {
    checkState(T_TPDATAHEADER, "tpDataHeader");
        
    sdmDataObject_.valid_ = true;
    
    sdmDataObject_.startTime_ = startTime;
    sdmDataObject_.dataOID_ = uid_;
    sdmDataObject_.title_ = title_;
    sdmDataObject_.dimensionality_ = 0;
    sdmDataObject_.numTime_ = numberOfIntegrations;
    sdmDataObject_.execBlockUID_ = execBlockUID;
    sdmDataObject_.execBlockNum_ = execBlockNum;
    sdmDataObject_.scanNum_ = scanNum;
    sdmDataObject_.subscanNum_ = subscanNum;
    sdmDataObject_.numAntenna_ = numAntenna;
    sdmDataObject_.correlationMode_ = AUTO_ONLY;
    sdmDataObject_.spectralResolutionType_ = BASEBAND_WIDE;
    sdmDataObject_.dataStruct_ = dataStruct;

    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Transfer-Encoding: 8bit");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();
    
    outputln(sdmDataObject_.toXML());
    outputln("--"+MIMEBOUNDARY_1);
  }
  
  void SDMDataObjectWriter::addTPSubscan(unsigned long long time,
					 unsigned long long interval,
					 const vector<unsigned long>& flags,
					 const vector<long long>& actualTimes,
					 const vector<long long>& actualDurations,
					 const vector<float>& autoData){
    checkState(T_ADDTPSUBSCAN, "addTPSubscan");
    outputln("Content-Type: Multipart/Related; boundary=\""+MIMEBOUNDARY_2+"\";type=\"text/xml\"; start=\"<DataSubset.xml>\"");
    outputln("Content-Description: Data and metadata subset");
    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();

    SDMDataSubset tpDataSubset(&sdmDataObject_,
			       time,
			       interval,
			       autoData);

    tpDataSubset.flags_ = (tpDataSubset.nFlags_ = flags.size()) ? &tpDataSubset.flags_[0] : 0;
    tpDataSubset.actualTimes_ = (tpDataSubset.nActualTimes_ = actualTimes.size()) ? &tpDataSubset.actualTimes_[0] : 0;    
    tpDataSubset.actualDurations_ = (tpDataSubset.nActualDurations_ = actualDurations.size()) ? &tpDataSubset.actualDurations_[0] : 0;    
    outputln(tpDataSubset.toXML());
    //outputln();

    if (flags.size() != 0) {
      unsigned int numFlags = sdmDataObject_.dataStruct_.flags_.size();
      if (numFlags!=0 && numFlags != flags.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'flags' ("
	    << flags.size()
	    << "), is not equal to the number declared in the global header ("
	    << numFlags << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputln("Content-Location: " + tpDataSubset.projectPath() + "flags.bin");
      outputln();
      outputln<unsigned long>(flags);
    }
    
    if (actualTimes.size() != 0) {
      unsigned int numActualTimes = sdmDataObject_.dataStruct_.actualTimes_.size();
      if (numActualTimes != 0 && numActualTimes != actualTimes.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'actualTimes' ("
	    << actualTimes.size()
	    << "), is not equal to the number declared in the global header ("
	    << numActualTimes << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputln("Content-Location: " + tpDataSubset.projectPath() + "actualTimes.bin");

      outputln();
      outputln<long long>(actualTimes);
    }

    if (actualDurations.size() != 0) {
      unsigned int numActualDurations = sdmDataObject_.dataStruct_.actualDurations_.size();
      if (numActualDurations != 0 && numActualDurations != actualDurations.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'actualDurations' ("
	    << actualDurations.size()
	    << "), is not equal to the number declared in the global header ("
	    << numActualDurations << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputln("Content-Location: " + tpDataSubset.projectPath() + "actualDurations.bin");
      outputln();
      outputln<long long>(actualDurations);
    }

    unsigned int numAutoData = sdmDataObject_.dataStruct_.autoData_.size();
    if (numAutoData != 0 && numAutoData != autoData.size()) {
      ostringstream oss;
      oss << "The number of values provided for 'autoData' ("
	  << autoData.size()
	  << "), is not equal to the number declared in the global header ("
	  << numAutoData << ").";
      throw SDMDataObjectWriterException(oss.str());
    }
    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: binary/octet-stream");
    outputln("Content-Location: " + tpDataSubset.projectPath() + "autoData.bin");
    outputln();
    outputln<float>(autoData);
    outputln("--"+MIMEBOUNDARY_2+"--");
  }

  void SDMDataObjectWriter::tpData(unsigned long long int startTime,
				   const string& execBlockUID,
				   unsigned int execBlockNum,
				   unsigned int scanNum,
				   unsigned int subscanNum,
				   unsigned int numberOfIntegrations,
				   unsigned int numAntenna,
				   const vector<SDMDataObject::Baseband>& basebands,
				   unsigned long long time,
				   unsigned long long interval,
				   const vector<AxisName>& autoDataAxes,
				   const vector<float>& autoData) {
    checkState(T_TPDATA, "tpData");

    SDMDataObject::DataStruct dataStruct;
    dataStruct.basebands_ = basebands;
    dataStruct.autoData_  = SDMDataObject::BinaryPart(autoData.size(), autoDataAxes);
    
    sdmDataObject_.valid_ = true;

    sdmDataObject_.startTime_ = startTime;
    sdmDataObject_.dataOID_ = uid_;
    sdmDataObject_.title_ = title_;
    sdmDataObject_.dimensionality_ = 0;
    sdmDataObject_.numTime_ = numberOfIntegrations;
    sdmDataObject_.execBlockUID_ = execBlockUID;
    sdmDataObject_.execBlockNum_ = execBlockNum;
    sdmDataObject_.scanNum_ = scanNum;
    sdmDataObject_.subscanNum_ = subscanNum;
    sdmDataObject_.numAntenna_ = numAntenna;
    sdmDataObject_.correlationMode_ = AUTO_ONLY;
    sdmDataObject_.spectralResolutionType_ = BASEBAND_WIDE;
    sdmDataObject_.dataStruct_ = dataStruct;

    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Transfer-Encoding: 8bit");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();
        
    outputln(sdmDataObject_.toXML());
    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: Multipart/Related; boundary=\""+MIMEBOUNDARY_2+"\";type=\"text/xml\"; start=\"<DataSubset.xml>\"");
    outputln("Content-Description: Data and metadata subset");
    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();

    SDMDataSubset tpDataSubset(&sdmDataObject_,
			       time,
			       interval,
			       autoData);
    outputln(tpDataSubset.toXML());
    //outputln();

    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: binary/octet-stream");
    outputln("Content-Location: " + tpDataSubset.projectPath() + "autoData.bin");
    outputln();
    
    outputln<float>(autoData);
    outputln("--"+MIMEBOUNDARY_2+"--");
  }

  void SDMDataObjectWriter::tpData(unsigned long long int startTime,
				   const string& execBlockUID,
				   unsigned int execBlockNum,
				   unsigned int scanNum,
				   unsigned int subscanNum,
				   unsigned int numberOfIntegrations,
				   unsigned int numAntenna,

				   const vector<SDMDataObject::Baseband>& basebands,

				   unsigned long long time,
				   unsigned long long interval,

				   const vector<AxisName>& flagsAxes,
				   const vector<unsigned long>& flags,
				   const vector<AxisName>& actualTimesAxes,
				   const vector<long long>& actualTimes,
				   const vector<AxisName>& actualDurationsAxes,
				   const vector<long long>& actualDurations,
				   const vector<AxisName>& autoDataAxes,
				   const vector<float>& autoData) {
    checkState(T_TPDATA, "tpData");
    
    SDMDataObject::DataStruct dataStruct;
    dataStruct.basebands_ = basebands;
    if (flags.size()) dataStruct.flags_ = SDMDataObject::BinaryPart(flags.size(), flagsAxes);
    if (actualTimes.size()) dataStruct.actualTimes_ = SDMDataObject::BinaryPart(actualTimes.size(), actualTimesAxes);
    if (actualDurations.size()) dataStruct.actualDurations_ = SDMDataObject::BinaryPart(actualDurations.size(), actualDurationsAxes);
    dataStruct.autoData_  = SDMDataObject::BinaryPart(autoData.size(), autoDataAxes);
    
    sdmDataObject_.valid_ = true;

    sdmDataObject_.startTime_ = startTime;
    sdmDataObject_.dataOID_ = uid_;
    sdmDataObject_.title_ = title_;
    sdmDataObject_.dimensionality_ = 0;
    sdmDataObject_.numTime_ = numberOfIntegrations;
    sdmDataObject_.execBlockUID_ = execBlockUID;
    sdmDataObject_.execBlockNum_ = execBlockNum;
    sdmDataObject_.scanNum_ = scanNum;
    sdmDataObject_.subscanNum_ = subscanNum;
    sdmDataObject_.numAntenna_ = numAntenna;
    sdmDataObject_.correlationMode_ = AUTO_ONLY;
    sdmDataObject_.spectralResolutionType_ = BASEBAND_WIDE;
    sdmDataObject_.dataStruct_ = dataStruct;

    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Transfer-Encoding: 8bit");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();
    
    outputln(sdmDataObject_.toXML());
    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: Multipart/Related; boundary=\""+MIMEBOUNDARY_2+"\";type=\"text/xml\"; start=\"<DataSubset.xml>\"");
    outputln("Content-Description: Data and metadata subset");
    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();

    SDMDataSubset tpDataSubset(&sdmDataObject_,
			       time,
			       interval,
			       autoData);

    tpDataSubset.flags_ = (tpDataSubset.nFlags_ = flags.size()) ? &tpDataSubset.flags_[0] : 0;
    tpDataSubset.actualTimes_ = (tpDataSubset.nActualTimes_ = actualTimes.size()) ? &tpDataSubset.actualTimes_[0] : 0;    
    tpDataSubset.actualDurations_ = (tpDataSubset.nActualDurations_ = actualDurations.size()) ? &tpDataSubset.actualDurations_[0] : 0;    
    outputln(tpDataSubset.toXML());
    //outputln();



    if (flags.size() != 0) {
      unsigned int numFlags = sdmDataObject_.dataStruct_.flags_.size();
      if (numFlags!=0 && numFlags != flags.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'flags' ("
	    << flags.size()
	    << "), is not equal to the number declared in the global header ("
	    << numFlags << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputln("Content-Location: " + tpDataSubset.projectPath() + "flags.bin");
      outputln();
      outputln<unsigned long>(flags);
    }
    
    if (actualTimes.size() != 0) {
      unsigned int numActualTimes = sdmDataObject_.dataStruct_.actualTimes_.size();
      if (numActualTimes != 0 && numActualTimes != actualTimes.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'actualTimes' ("
	    << actualTimes.size()
	    << "), is not equal to the number declared in the global header ("
	    << numActualTimes << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputln("Content-Location: " + tpDataSubset.projectPath() + "actualTimes.bin");

      outputln();
      outputln<long long>(actualTimes);
    }

    if (actualDurations.size() != 0) {
      unsigned int numActualDurations = sdmDataObject_.dataStruct_.actualDurations_.size();
      if (numActualDurations != 0 && numActualDurations != actualDurations.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'actualDurations' ("
	    << actualDurations.size()
	    << "), is not equal to the number declared in the global header ("
	    << numActualDurations << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputln("Content-Location: " + tpDataSubset.projectPath() + "actualDurations.bin");
      outputln();
      outputln<long long>(actualDurations);
    }

    unsigned int numAutoData = sdmDataObject_.dataStruct_.autoData_.size();
    if (numAutoData != 0 && numAutoData != autoData.size()) {
      ostringstream oss;
      oss << "The number of values provided for 'autoData' ("
	  << autoData.size()
	  << "), is not equal to the number declared in the global header ("
	  << numAutoData << ").";
      throw SDMDataObjectWriterException(oss.str());
    }
    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: binary/octet-stream");
    outputln("Content-Location: " + tpDataSubset.projectPath() + "autoData.bin");
    outputln();
    outputln<float>(autoData);
    outputln("--"+MIMEBOUNDARY_2+"--");
  }
  
  /**
   * Writes the XML global header into its attachment on the MIME message stream.
   */
  void SDMDataObjectWriter::corrDataHeader(unsigned long long startTime,
					   const string& execBlockUID,
					   unsigned int execBlockNum,
					   unsigned int scanNum,
					   unsigned int subscanNum,
					   unsigned int numAntenna,
					   CorrelationMode correlationMode,
					   SpectralResolutionType spectralResolutionType,
					   SDMDataObject::DataStruct& dataStruct) {
    checkState(T_CORRDATAHEADER, "corrDataHeader");
    
    ostringstream oss;
    oss << "/" << execBlockNum << "/" << scanNum << "/" << subscanNum;
    subscanPath_ = oss.str();
    
    sdmDataObject_.valid_ = true;

    sdmDataObject_.title_ = title_;
    sdmDataObject_.startTime_ = startTime;
    sdmDataObject_.dataOID_ = uid_;
    sdmDataObject_.dimensionality_ = 1;
    sdmDataObject_.numTime_ = 0;
    sdmDataObject_.execBlockUID_ = execBlockUID;
    sdmDataObject_.execBlockNum_ = execBlockNum;
    sdmDataObject_.scanNum_ = scanNum;
    sdmDataObject_.subscanNum_ = subscanNum;
    sdmDataObject_.numAntenna_ = numAntenna;
    sdmDataObject_.correlationMode_ = correlationMode;
    sdmDataObject_.spectralResolutionType_ = spectralResolutionType;
    sdmDataObject_.dataStruct_ = dataStruct;
    
    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Transfer-Encoding: 8bit");
    outputln("Content-Location: " + sdmDataObject_.projectPath() + "desc.xml");
    outputln();

    outputln(sdmDataObject_.toXML());
  }
  

  void SDMDataObjectWriter::addData(unsigned int integrationNum,
				    unsigned int subintegrationNum,
				    unsigned long long time,
				    unsigned long long interval,
				    const vector<unsigned long>& flags,
				    const vector<long long>& actualTimes,
				    const vector<long long>& actualDurations,
				    const vector<float>& zeroLags,
				    const vector<int>& longCrossData,
				    const vector<short>& shortCrossData,
				    const vector<float>& floatCrossData,
				    const vector<float>& autoData) {
    SDMDataSubset sdmDataSubset(&sdmDataObject_);
    sdmDataObject_.numTime_++;
    sdmDataSubsetNum_++;
    
    // integrationNum and subintegrationNum.
    sdmDataSubset.integrationNum_    = integrationNum;
    if (sdmDataSubset.owner_->spectralResolutionType_ == CHANNEL_AVERAGE) 
      sdmDataSubset.subintegrationNum_ = subintegrationNum;

    // The time.
    sdmDataSubset.time_ = time;

    // The interval.
    sdmDataSubset.interval_ = interval;

    // The crossDataType.
    if (longCrossData.size() != 0) 
      sdmDataSubset.crossDataType_ = INT_TYPE;

    else if (shortCrossData.size() != 0)
      sdmDataSubset.crossDataType_ = SHORT_TYPE;

    else if (floatCrossData.size() != 0) 
      sdmDataSubset.crossDataType_ = FLOAT_TYPE;

    // Attachments size;
    sdmDataSubset.nActualTimes_     = actualTimes.size();
    sdmDataSubset.nActualDurations_ = actualDurations.size();
    sdmDataSubset.nZeroLags_        = zeroLags.size();
    sdmDataSubset.nFlags_   = flags.size();
    sdmDataSubset.nFlags_   = flags.size();
    switch (sdmDataSubset.crossDataType_) {
    case INT_TYPE:
      sdmDataSubset.nCrossData_ = longCrossData.size();
      break;
    case SHORT_TYPE:
      sdmDataSubset.nCrossData_ = shortCrossData.size();
      break;
    case FLOAT_TYPE:
      sdmDataSubset.nCrossData_ = floatCrossData.size();
      break;
    default:
      sdmDataSubset.nCrossData_ = 0;
    }

    //sdmDataSubset.nCrossData_       = shortCrossData.size() ? shortCrossData.size():longCrossData.size();

    sdmDataSubset.nAutoData_        = autoData.size();

    outputln("--"+MIMEBOUNDARY_1);
    outputln("Content-Type: Multipart/Related; boundary=\""+MIMEBOUNDARY_2+"\";type=\"text/xml\"");
    outputln("Content-Description: Data and metadata subset");
    outputln("--"+MIMEBOUNDARY_2);
    outputln("Content-Type: text/xml; charset=\"UTF-8\"");
    outputln("Content-Location: " + sdmDataSubset.projectPath() + "desc.xml");
    outputln();
    outputln(sdmDataSubset.toXML());
    

    if (flags.size() != 0) {
      unsigned int numFlags = sdmDataObject_.dataStruct_.flags_.size();
      if (numFlags !=0 && numFlags != flags.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'flags' ("
	    << flags.size()
	    << "), is not equal to the number declared in the global header ("
	    << numFlags << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputlnLocation("flags", sdmDataSubset);
      outputln();
      outputln<unsigned long>(flags);
    }
    

    if (actualTimes.size() != 0) {
      unsigned int numActualTimes = sdmDataObject_.dataStruct_.actualTimes_.size();
      if (numActualTimes != 0 && numActualTimes != actualTimes.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'actualTimes' ("
	    << actualTimes.size()
	    << "), is not equal to the number declared in the global header ("
	    << numActualTimes << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputlnLocation("actualTimes", sdmDataSubset);
      outputln();
      outputln<long long>(actualTimes);
    }

    if (actualDurations.size() != 0) {
      unsigned int numActualDurations = sdmDataObject_.dataStruct_.actualDurations_.size();
      if (numActualDurations != 0 && numActualDurations != actualDurations.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'actualDurations' ("
	    << actualDurations.size()
	    << "), is not equal to the number declared in the global header ("
	    << numActualDurations << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputlnLocation("actualDurations", sdmDataSubset);
      outputln();
      outputln<long long>(actualDurations);
    }
    
    
    if (sdmDataObject_.correlationMode_ != AUTO_ONLY) {
      int numCrossData = sdmDataObject_.dataStruct_.crossData_.size();
      int numCrossDataV = 0; //= longCrossData.size() ? longCrossData.size():shortCrossData.size();
      switch(sdmDataSubset.crossDataType_) {
      case INT_TYPE:
	numCrossDataV = longCrossData.size();
	break;
      case SHORT_TYPE:
	numCrossDataV = shortCrossData.size();
	break;
      case FLOAT_TYPE:
	numCrossDataV = floatCrossData.size();
	break;
      default:
	break;
      }
      if (numCrossData != numCrossDataV) {
	ostringstream oss;
	oss << "The number of values provided for 'crossData' ("
	    << numCrossDataV
	    << "), is not equal to the number declared in the global header ("
	    << numCrossData << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputlnLocation("crossData", sdmDataSubset);
      outputln();
      switch (sdmDataSubset.crossDataType_) {
      case INT_TYPE:
	outputln<int>(longCrossData);
	break;
      case SHORT_TYPE:
	outputln<short>(shortCrossData);
	break;
      case FLOAT_TYPE:
	outputln<float>(floatCrossData);
	break;
      default: 
	throw SDMDataObjectWriterException("'" + CPrimitiveDataType::name(sdmDataSubset.crossDataType_)+"' data are not processed here."); 
      }
    }

    if (sdmDataObject_.correlationMode_ !=  CROSS_ONLY) {
      unsigned int numAutoData = sdmDataObject_.dataStruct_.autoData_.size();
      if (numAutoData != autoData.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'autoData' ("
	    << autoData.size()
	    << "), is not equal to the number declared in the global header ("
	    << numAutoData << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputlnLocation("autoData", sdmDataSubset);
      outputln();
      outputln<float>(autoData);
    }
    
    // if (sdmDataObject_.spectralResolutionType_ != CHANNEL_AVERAGE) {  
    // Now the zeroLags are optionally allowed in CHANNEL_AVERAGE
    // zeroLags are now optional in any case - Michel Caillat - 24 Jul 2008
    unsigned int numZeroLags = sdmDataObject_.dataStruct_.zeroLags_.size();
    if (numZeroLags > 0) {  
      if (numZeroLags != zeroLags.size()) {
	ostringstream oss;
	oss << "The number of values provided for 'zeroLags' ("
	    << zeroLags.size()
	    << "), is not equal to the number declared in the global header ("
	    << numZeroLags << ").";
	throw SDMDataObjectWriterException(oss.str());
      }
      outputln("--"+MIMEBOUNDARY_2);
      outputln("Content-Type: binary/octet-stream");
      outputlnLocation("zeroLags", sdmDataSubset);
      outputln();
      outputln<float>(zeroLags);
    }

    outputln("--"+MIMEBOUNDARY_2+"--");
  }

  void SDMDataObjectWriter::addIntegration(unsigned int integrationNum,
					   unsigned long long time,
					   unsigned long long interval,
					   const vector<unsigned long>& flags,
					   const vector<long long>& actualTimes,
					   const vector<long long>& actualDurations,
					   const vector<float>& zeroLags,
					   const vector<int>& crossData,
					   const vector<float>& autoData) {

    checkState(T_ADDINTEGRATION, "addIntegration");

    vector<short> emptyShort;
    vector<float> emptyFloat;
    addData(integrationNum,
	    0,
	    time,
	    interval,
	    flags,
	    actualTimes,
	    actualDurations,
	    zeroLags,
	    crossData,
	    emptyShort,
	    emptyFloat,
	    autoData);		
  }
  

  void SDMDataObjectWriter::addIntegration(unsigned int integrationNum,
					   unsigned long long time,
					   unsigned long long interval,
					   const vector<unsigned long>& flags,
					   const vector<long long>& actualTimes,
					   const vector<long long>& actualDurations,
					   const vector<float>& zeroLags,
					   const vector<short>& crossData,
					   const vector<float>& autoData) {
    checkState(T_ADDINTEGRATION, "addIntegration");

    vector<int> emptyLong;
    vector<float> emptyFloat;
    addData(integrationNum,
	    0,
	    time,
	    interval,
	    flags,
	    actualTimes,
	    actualDurations,
	    zeroLags,
	    emptyLong,
	    crossData,
	    emptyFloat,
	    autoData);		
  }

  void SDMDataObjectWriter::addIntegration(unsigned int integrationNum,
					   unsigned long long time,
					   unsigned long long interval,
					   const vector<unsigned long>& flags,
					   const vector<long long>& actualTimes,
					   const vector<long long>& actualDurations,
					   const vector<float>& zeroLags,
					   const vector<float>& crossData,
					   const vector<float>& autoData) {
    checkState(T_ADDINTEGRATION, "addIntegration");

    vector<int> emptyLong;
    vector<short> emptyShort;
    addData(integrationNum,
	    0,
	    time,
	    interval,
	    flags,
	    actualTimes,
	    actualDurations,
	    zeroLags,
	    emptyLong,
	    emptyShort,
	    crossData,
	    autoData);		
  }

  void SDMDataObjectWriter::addSubintegration(unsigned int integrationNum,
					      unsigned int subIntegrationNum,
					      unsigned long long time,
					      unsigned long long interval,
					      const vector<unsigned long>& flags,
					      const vector<long long>& actualTimes,
					      const vector<long long>& actualDurations,
					      const vector<float>& zeroLags,
					      const vector<short>& crossData,
					      const vector<float>& autoData) {
    checkState(T_ADDSUBINTEGRATION, "addSubintegration");

    vector<int> emptyLong;
    vector<float> emptyFloat;
    addData(integrationNum,
	    subIntegrationNum,
	    time,
	    interval,
	    flags,
	    actualTimes,
	    actualDurations,
	    zeroLags,
	    emptyLong,
	    crossData,
	    emptyFloat,
	    autoData);		
  }

  void SDMDataObjectWriter::addSubintegration(unsigned int integrationNum,
					      unsigned int subIntegrationNum,
					      unsigned long long time,
					      unsigned long long interval,
					      const vector<unsigned long>& flags,
					      const vector<long long>& actualTimes,
					      const vector<long long>& actualDurations,
					      const vector<float>& zeroLags,
					      const vector<int>& crossData,
					      const vector<float>& autoData) {
    checkState(T_ADDSUBINTEGRATION, "addSubIntegration");

    vector<short> emptyShort;
    vector<float> emptyFloat;
    addData(integrationNum,
	    subIntegrationNum,
	    time,
	    interval,
	    flags,
	    actualTimes,
	    actualDurations,
	    zeroLags,
	    crossData,
	    emptyShort,
	    emptyFloat,
	    autoData);		
  }

  void SDMDataObjectWriter::addSubintegration(unsigned int integrationNum,
					      unsigned int subIntegrationNum,
					      unsigned long long time,
					      unsigned long long interval,
					      const vector<unsigned long>& flags,
					      const vector<long long>& actualTimes,
					      const vector<long long>& actualDurations,
					      const vector<float>& zeroLags,
					      const vector<float>& crossData,
					      const vector<float>& autoData) {
    checkState(T_ADDSUBINTEGRATION, "addSubIntegration");

    vector<int> emptyLong;
    vector<short> emptyShort;
    addData(integrationNum,
	    subIntegrationNum,
	    time,
	    interval,
	    flags,
	    actualTimes,
	    actualDurations,
	    zeroLags,
	    emptyLong,
	    emptyShort,
	    crossData,
	    autoData);		
  }
  
  
  bool SDMDataObjectWriter::initClass() {
    return true;
  }

  void SDMDataObjectWriter:: checkState(Transitions t, const string& methodName) {
    switch(currentState_) {
    case START:
      if (t == T_TPDATA ) { 
	currentState_ = S_TPDATA;
	return;
      }
      else if (t == T_TPDATAHEADER ) {
	currentState_ = S_TPDATAHEADER;
	return;
      }
      else if (t == T_CORRDATAHEADER) {
	currentState_ = S_CORRDATAHEADER;
	return;
      }
      break;

    case S_TPDATA:
      if (t == T_DONE) {
	currentState_ = END;
	return;
      }
      break;
      
    case S_TPDATAHEADER:
      if (t == T_ADDTPSUBSCAN) {
	currentState_ = S_ADDTPSUBSCAN;
	return;
      }
      break;

    case S_ADDTPSUBSCAN:
      if ( t == T_DONE ) {
	currentState_ = END;
	return;
      }
      break;

    case S_CORRDATAHEADER:
      if (t == T_ADDINTEGRATION) {
	currentState_ = S_ADDINTEGRATION;
	return;
      }
      else if (t == T_ADDSUBINTEGRATION) {
	currentState_ = S_ADDSUBINTEGRATION;
	return;
      }
      break;
      
    case S_ADDINTEGRATION:
      if (t == T_ADDINTEGRATION)
	return;
      else if (t == T_DONE) {
	currentState_ = END;
	return;
      }
      break;

      
    case S_ADDSUBINTEGRATION:
      if (t == T_ADDSUBINTEGRATION)
	return;
      else if (t == T_DONE) {
	currentState_ = END;
	return;
      }
      break;

	
    case END:
      break;
    }
    throw SDMDataObjectWriterException("Invalid call of method '" + methodName + "'");
  }
}


