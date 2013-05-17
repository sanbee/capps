
/*
 * ALMA - Atacama Large Millimeter Array
 * (c) European Southern Observatory, 2002
 * (c) Associated Universities Inc., 2002
 * Copyright by ESO (in the framework of the ALMA collaboration),
 * Copyright by AUI (in the framework of the ALMA collaboration),
 * All rights reserved.
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY, without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307  USA
 *
 * Warning!
 *  -------------------------------------------------------------------- 
 * | This is generated code!  Do not modify this file.                  |
 * | If you do, all changes will be lost when the file is re-generated. |
 *  --------------------------------------------------------------------
 *
 * File ConfigDescriptionRow.h
 */
 
#ifndef ConfigDescriptionRow_CLASS
#define ConfigDescriptionRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::ConfigDescriptionRowIDL;
#endif

#include <Angle.h>
#include <AngularRate.h>
#include <ArrayTime.h>
#include <ArrayTimeInterval.h>
#include <Complex.h>
#include <Entity.h>
#include <EntityId.h>
#include <EntityRef.h>
#include <Flux.h>
#include <Frequency.h>
#include <Humidity.h>
#include <Interval.h>
#include <Length.h>
#include <Pressure.h>
#include <Speed.h>
#include <Tag.h>
#include <Temperature.h>
#include <ConversionException.h>
#include <NoSuchRow.h>
#include <IllegalAccessException.h>

/*
#include <Enumerations.h>
using namespace enumerations;
 */




	

	

	

	

	
#include "CCorrelationMode.h"
using namespace CorrelationModeMod;
	

	

	

	
#include "CAtmPhaseCorrection.h"
using namespace AtmPhaseCorrectionMod;
	

	
#include "CSpectralResolutionType.h"
using namespace SpectralResolutionTypeMod;
	



using asdm::Angle;
using asdm::AngularRate;
using asdm::ArrayTime;
using asdm::Complex;
using asdm::Entity;
using asdm::EntityId;
using asdm::EntityRef;
using asdm::Flux;
using asdm::Frequency;
using asdm::Humidity;
using asdm::Interval;
using asdm::Length;
using asdm::Pressure;
using asdm::Speed;
using asdm::Tag;
using asdm::Temperature;
using asdm::ConversionException;
using asdm::NoSuchRow;
using asdm::IllegalAccessException;

/*\file ConfigDescription.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::ConfigDescriptionTable;


// class asdm::ProcessorRow;
class ProcessorRow;

// class asdm::AntennaRow;
class AntennaRow;

// class asdm::DataDescriptionRow;
class DataDescriptionRow;

// class asdm::SwitchCycleRow;
class SwitchCycleRow;

// class asdm::FeedRow;
class FeedRow;

// class asdm::ConfigDescriptionRow;
class ConfigDescriptionRow;
	

/**
 * The ConfigDescriptionRow class is a row of a ConfigDescriptionTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class ConfigDescriptionRow {
friend class asdm::ConfigDescriptionTable;

public:

	virtual ~ConfigDescriptionRow();

	/**
	 * Return the table to which this row belongs.
	 */
	ConfigDescriptionTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a ConfigDescriptionRowIDL struct.
	 */
	ConfigDescriptionRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct ConfigDescriptionRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (ConfigDescriptionRowIDL x) throw(ConversionException);
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string toXML() const;

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void setFromXML (string rowDoc) throw(ConversionException);
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute numAntenna
	
	
	

	
 	/**
 	 * Get numAntenna.
 	 * @return numAntenna as int
 	 */
 	int getNumAntenna() const;
	
 
 	
 	
 	/**
 	 * Set numAntenna with the specified int.
 	 * @param numAntenna The int value to which numAntenna is to be set.
 	 
 		
 			
 	 */
 	void setNumAntenna (int numAntenna);
  		
	
	
	


	
	// ===> Attribute numFeed
	
	
	

	
 	/**
 	 * Get numFeed.
 	 * @return numFeed as int
 	 */
 	int getNumFeed() const;
	
 
 	
 	
 	/**
 	 * Set numFeed with the specified int.
 	 * @param numFeed The int value to which numFeed is to be set.
 	 
 		
 			
 	 */
 	void setNumFeed (int numFeed);
  		
	
	
	


	
	// ===> Attribute numSubBand
	
	
	

	
 	/**
 	 * Get numSubBand.
 	 * @return numSubBand as vector<int >
 	 */
 	vector<int > getNumSubBand() const;
	
 
 	
 	
 	/**
 	 * Set numSubBand with the specified vector<int >.
 	 * @param numSubBand The vector<int > value to which numSubBand is to be set.
 	 
 		
 			
 	 */
 	void setNumSubBand (vector<int > numSubBand);
  		
	
	
	


	
	// ===> Attribute phasedArrayList, which is optional
	
	
	
	/**
	 * The attribute phasedArrayList is optional. Return true if this attribute exists.
	 * @return true if and only if the phasedArrayList attribute exists. 
	 */
	bool isPhasedArrayListExists() const;
	

	
 	/**
 	 * Get phasedArrayList, which is optional.
 	 * @return phasedArrayList as vector<int >
 	 * @throws IllegalAccessException If phasedArrayList does not exist.
 	 */
 	vector<int > getPhasedArrayList() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set phasedArrayList with the specified vector<int >.
 	 * @param phasedArrayList The vector<int > value to which phasedArrayList is to be set.
 	 
 		
 	 */
 	void setPhasedArrayList (vector<int > phasedArrayList);
		
	
	
	
	/**
	 * Mark phasedArrayList, which is an optional field, as non-existent.
	 */
	void clearPhasedArrayList ();
	


	
	// ===> Attribute correlationMode
	
	
	

	
 	/**
 	 * Get correlationMode.
 	 * @return correlationMode as CorrelationModeMod::CorrelationMode
 	 */
 	CorrelationModeMod::CorrelationMode getCorrelationMode() const;
	
 
 	
 	
 	/**
 	 * Set correlationMode with the specified CorrelationModeMod::CorrelationMode.
 	 * @param correlationMode The CorrelationModeMod::CorrelationMode value to which correlationMode is to be set.
 	 
 		
 			
 	 */
 	void setCorrelationMode (CorrelationModeMod::CorrelationMode correlationMode);
  		
	
	
	


	
	// ===> Attribute flagAnt, which is optional
	
	
	
	/**
	 * The attribute flagAnt is optional. Return true if this attribute exists.
	 * @return true if and only if the flagAnt attribute exists. 
	 */
	bool isFlagAntExists() const;
	

	
 	/**
 	 * Get flagAnt, which is optional.
 	 * @return flagAnt as vector<bool >
 	 * @throws IllegalAccessException If flagAnt does not exist.
 	 */
 	vector<bool > getFlagAnt() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set flagAnt with the specified vector<bool >.
 	 * @param flagAnt The vector<bool > value to which flagAnt is to be set.
 	 
 		
 	 */
 	void setFlagAnt (vector<bool > flagAnt);
		
	
	
	
	/**
	 * Mark flagAnt, which is an optional field, as non-existent.
	 */
	void clearFlagAnt ();
	


	
	// ===> Attribute configDescriptionId
	
	
	

	
 	/**
 	 * Get configDescriptionId.
 	 * @return configDescriptionId as Tag
 	 */
 	Tag getConfigDescriptionId() const;
	
 
 	
 	
	
	


	
	// ===> Attribute atmPhaseCorrection
	
	
	

	
 	/**
 	 * Get atmPhaseCorrection.
 	 * @return atmPhaseCorrection as vector<AtmPhaseCorrectionMod::AtmPhaseCorrection >
 	 */
 	vector<AtmPhaseCorrectionMod::AtmPhaseCorrection > getAtmPhaseCorrection() const;
	
 
 	
 	
 	/**
 	 * Set atmPhaseCorrection with the specified vector<AtmPhaseCorrectionMod::AtmPhaseCorrection >.
 	 * @param atmPhaseCorrection The vector<AtmPhaseCorrectionMod::AtmPhaseCorrection > value to which atmPhaseCorrection is to be set.
 	 
 		
 			
 	 */
 	void setAtmPhaseCorrection (vector<AtmPhaseCorrectionMod::AtmPhaseCorrection > atmPhaseCorrection);
  		
	
	
	


	
	// ===> Attribute assocNature, which is optional
	
	
	
	/**
	 * The attribute assocNature is optional. Return true if this attribute exists.
	 * @return true if and only if the assocNature attribute exists. 
	 */
	bool isAssocNatureExists() const;
	

	
 	/**
 	 * Get assocNature, which is optional.
 	 * @return assocNature as vector<SpectralResolutionTypeMod::SpectralResolutionType >
 	 * @throws IllegalAccessException If assocNature does not exist.
 	 */
 	vector<SpectralResolutionTypeMod::SpectralResolutionType > getAssocNature() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set assocNature with the specified vector<SpectralResolutionTypeMod::SpectralResolutionType >.
 	 * @param assocNature The vector<SpectralResolutionTypeMod::SpectralResolutionType > value to which assocNature is to be set.
 	 
 		
 	 */
 	void setAssocNature (vector<SpectralResolutionTypeMod::SpectralResolutionType > assocNature);
		
	
	
	
	/**
	 * Mark assocNature, which is an optional field, as non-existent.
	 */
	void clearAssocNature ();
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId
	
	
	

	
 	/**
 	 * Get antennaId.
 	 * @return antennaId as vector<Tag> 
 	 */
 	vector<Tag>  getAntennaId() const;
	
 
 	
 	
 	/**
 	 * Set antennaId with the specified vector<Tag> .
 	 * @param antennaId The vector<Tag>  value to which antennaId is to be set.
 	 
 		
 			
 	 */
 	void setAntennaId (vector<Tag>  antennaId);
  		
	
	
	


	
	// ===> Attribute assocConfigDescriptionId, which is optional
	
	
	
	/**
	 * The attribute assocConfigDescriptionId is optional. Return true if this attribute exists.
	 * @return true if and only if the assocConfigDescriptionId attribute exists. 
	 */
	bool isAssocConfigDescriptionIdExists() const;
	

	
 	/**
 	 * Get assocConfigDescriptionId, which is optional.
 	 * @return assocConfigDescriptionId as vector<Tag> 
 	 * @throws IllegalAccessException If assocConfigDescriptionId does not exist.
 	 */
 	vector<Tag>  getAssocConfigDescriptionId() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set assocConfigDescriptionId with the specified vector<Tag> .
 	 * @param assocConfigDescriptionId The vector<Tag>  value to which assocConfigDescriptionId is to be set.
 	 
 		
 	 */
 	void setAssocConfigDescriptionId (vector<Tag>  assocConfigDescriptionId);
		
	
	
	
	/**
	 * Mark assocConfigDescriptionId, which is an optional field, as non-existent.
	 */
	void clearAssocConfigDescriptionId ();
	


	
	// ===> Attribute dataDescriptionId
	
	
	

	
 	/**
 	 * Get dataDescriptionId.
 	 * @return dataDescriptionId as vector<Tag> 
 	 */
 	vector<Tag>  getDataDescriptionId() const;
	
 
 	
 	
 	/**
 	 * Set dataDescriptionId with the specified vector<Tag> .
 	 * @param dataDescriptionId The vector<Tag>  value to which dataDescriptionId is to be set.
 	 
 		
 			
 	 */
 	void setDataDescriptionId (vector<Tag>  dataDescriptionId);
  		
	
	
	


	
	// ===> Attribute feedId
	
	
	

	
 	/**
 	 * Get feedId.
 	 * @return feedId as vector<int> 
 	 */
 	vector<int>  getFeedId() const;
	
 
 	
 	
 	/**
 	 * Set feedId with the specified vector<int> .
 	 * @param feedId The vector<int>  value to which feedId is to be set.
 	 
 		
 			
 	 */
 	void setFeedId (vector<int>  feedId);
  		
	
	
	


	
	// ===> Attribute processorId
	
	
	

	
 	/**
 	 * Get processorId.
 	 * @return processorId as Tag
 	 */
 	Tag getProcessorId() const;
	
 
 	
 	
 	/**
 	 * Set processorId with the specified Tag.
 	 * @param processorId The Tag value to which processorId is to be set.
 	 
 		
 			
 	 */
 	void setProcessorId (Tag processorId);
  		
	
	
	


	
	// ===> Attribute switchCycleId
	
	
	

	
 	/**
 	 * Get switchCycleId.
 	 * @return switchCycleId as vector<Tag> 
 	 */
 	vector<Tag>  getSwitchCycleId() const;
	
 
 	
 	
 	/**
 	 * Set switchCycleId with the specified vector<Tag> .
 	 * @param switchCycleId The vector<Tag>  value to which switchCycleId is to be set.
 	 
 		
 			
 	 */
 	void setSwitchCycleId (vector<Tag>  switchCycleId);
  		
	
	
	


	///////////
	// Links //
	///////////
	
	

	
		
	/**
	 * processorId pointer to the row in the Processor table having Processor.processorId == processorId
	 * @return a ProcessorRow*
	 * 
	 
	 */
	 ProcessorRow* getProcessorUsingProcessorId();
	 

	

	
 		
 	/**
 	 * Set antennaId[i] with the specified Tag.
 	 * @param i The index in antennaId where to set the Tag value.
 	 * @param antennaId The Tag value to which antennaId[i] is to be set. 
	 		
 	 * @throws IndexOutOfBoundsException
  	 */
  	void setAntennaId (int i, Tag antennaId); 
 			
	

	
		 
/**
 * Append a Tag to antennaId.
 * @param id the Tag to be appended to antennaId
 */
 void addAntennaId(Tag id); 

/**
 * Append a vector of Tag to antennaId.
 * @param id an array of Tag to be appended to antennaId
 */
 void addAntennaId(const vector<Tag> & id); 
 

 /**
  * Returns the Tag stored in antennaId at position i.
  * @param i the position in antennaId where the Tag is retrieved.
  * @return the Tag stored at position i in antennaId.
  */
 const Tag getAntennaId(int i);
 
 /**
  * Returns the AntennaRow linked to this row via the tag stored in antennaId
  * at position i.
  * @param i the position in antennaId.
  * @return a pointer on a AntennaRow whose key (a Tag) is equal to the Tag stored at position
  * i in the antennaId. 
  */
 AntennaRow* getAntenna(int i); 
 
 /**
  * Returns the vector of AntennaRow* linked to this row via the Tags stored in antennaId
  * @return an array of pointers on AntennaRow.
  */
 vector<AntennaRow *> getAntennas(); 
  

	

	
 		
 	/**
 	 * Set dataDescriptionId[i] with the specified Tag.
 	 * @param i The index in dataDescriptionId where to set the Tag value.
 	 * @param dataDescriptionId The Tag value to which dataDescriptionId[i] is to be set. 
	 		
 	 * @throws IndexOutOfBoundsException
  	 */
  	void setDataDescriptionId (int i, Tag dataDescriptionId); 
 			
	

	
		 
/**
 * Append a Tag to dataDescriptionId.
 * @param id the Tag to be appended to dataDescriptionId
 */
 void addDataDescriptionId(Tag id); 

/**
 * Append a vector of Tag to dataDescriptionId.
 * @param id an array of Tag to be appended to dataDescriptionId
 */
 void addDataDescriptionId(const vector<Tag> & id); 
 

 /**
  * Returns the Tag stored in dataDescriptionId at position i.
  * @param i the position in dataDescriptionId where the Tag is retrieved.
  * @return the Tag stored at position i in dataDescriptionId.
  */
 const Tag getDataDescriptionId(int i);
 
 /**
  * Returns the DataDescriptionRow linked to this row via the tag stored in dataDescriptionId
  * at position i.
  * @param i the position in dataDescriptionId.
  * @return a pointer on a DataDescriptionRow whose key (a Tag) is equal to the Tag stored at position
  * i in the dataDescriptionId. 
  */
 DataDescriptionRow* getDataDescription(int i); 
 
 /**
  * Returns the vector of DataDescriptionRow* linked to this row via the Tags stored in dataDescriptionId
  * @return an array of pointers on DataDescriptionRow.
  */
 vector<DataDescriptionRow *> getDataDescriptions(); 
  

	

	
 		
 	/**
 	 * Set switchCycleId[i] with the specified Tag.
 	 * @param i The index in switchCycleId where to set the Tag value.
 	 * @param switchCycleId The Tag value to which switchCycleId[i] is to be set. 
	 		
 	 * @throws IndexOutOfBoundsException
  	 */
  	void setSwitchCycleId (int i, Tag switchCycleId); 
 			
	

	
		 
/**
 * Append a Tag to switchCycleId.
 * @param id the Tag to be appended to switchCycleId
 */
 void addSwitchCycleId(Tag id); 

/**
 * Append a vector of Tag to switchCycleId.
 * @param id an array of Tag to be appended to switchCycleId
 */
 void addSwitchCycleId(const vector<Tag> & id); 
 

 /**
  * Returns the Tag stored in switchCycleId at position i.
  * @param i the position in switchCycleId where the Tag is retrieved.
  * @return the Tag stored at position i in switchCycleId.
  */
 const Tag getSwitchCycleId(int i);
 
 /**
  * Returns the SwitchCycleRow linked to this row via the tag stored in switchCycleId
  * at position i.
  * @param i the position in switchCycleId.
  * @return a pointer on a SwitchCycleRow whose key (a Tag) is equal to the Tag stored at position
  * i in the switchCycleId. 
  */
 SwitchCycleRow* getSwitchCycle(int i); 
 
 /**
  * Returns the vector of SwitchCycleRow* linked to this row via the Tags stored in switchCycleId
  * @return an array of pointers on SwitchCycleRow.
  */
 vector<SwitchCycleRow *> getSwitchCycles(); 
  

	

	
 		
 	/**
 	 * Set feedId[i] with the specified int.
 	 * @param i The index in feedId where to set the int value.
 	 * @param feedId The int value to which feedId[i] is to be set. 
	 		
 	 * @throws IndexOutOfBoundsException
  	 */
  	void setFeedId (int i, int feedId); 
 			
	

	
		

	// ===> Slices link from a row of ConfigDescription table to a collection of row of Feed table.	

	/**
	 * Append a new id to feedId
	 * @param id the int value to be appended to feedId
	 */
	void addFeedId(int id); 
	
	/**
	 * Append an array of ids to feedId
	 * @param id a vector of int containing the values to append to feedId.
	 */ 
	void addFeedId(vector<int> id); 


	/**
	 * Get the collection of rows in the Feed table having feedId == feedId[i]
	 * @return a vector of FeedRow *. 
	 */	 
	const vector <FeedRow *> getFeeds(int i);


	/** 
	 * Get the collection of rows in the Feed table having feedId == feedId[i]
	 * for any i in [O..feedId.size()-1].
	 * @return a vector of FeedRow *.
	 */
	const vector <FeedRow *> getFeeds();
	


	

	
 		
 	/**
 	 * Set assocConfigDescriptionId[i] with the specified Tag.
 	 * @param i The index in assocConfigDescriptionId where to set the Tag value.
 	 * @param assocConfigDescriptionId The Tag value to which assocConfigDescriptionId[i] is to be set. 
 	 * @throws OutOfBoundsException
  	 */
  	void setAssocConfigDescriptionId (int i, Tag assocConfigDescriptionId)  ;
 			
	

	
		 
/**
 * Append a Tag to assocConfigDescriptionId.
 * @param id the Tag to be appended to assocConfigDescriptionId
 */
 void addAssocConfigDescriptionId(Tag id); 

/**
 * Append a vector of Tag to assocConfigDescriptionId.
 * @param id an array of Tag to be appended to assocConfigDescriptionId
 */
 void addAssocConfigDescriptionId(const vector<Tag> & id); 
 

 /**
  * Returns the Tag stored in assocConfigDescriptionId at position i.
  * @param i the position in assocConfigDescriptionId where the Tag is retrieved.
  * @return the Tag stored at position i in assocConfigDescriptionId.
  */
 const Tag getAssocConfigDescriptionId(int i);
 
 /**
  * Returns the ConfigDescriptionRow linked to this row via the tag stored in assocConfigDescriptionId
  * at position i.
  * @param i the position in assocConfigDescriptionId.
  * @return a pointer on a ConfigDescriptionRow whose key (a Tag) is equal to the Tag stored at position
  * i in the assocConfigDescriptionId. 
  */
 ConfigDescriptionRow* getConfigDescription(int i); 
 
 /**
  * Returns the vector of ConfigDescriptionRow* linked to this row via the Tags stored in assocConfigDescriptionId
  * @return an array of pointers on ConfigDescriptionRow.
  */
 vector<ConfigDescriptionRow *> getConfigDescriptions(); 
  

	

	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this ConfigDescriptionRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(vector<Tag>  antennaId, vector<Tag>  dataDescriptionId, vector<int>  feedId, Tag processorId, vector<Tag>  switchCycleId, int numAntenna, int numFeed, vector<int > numSubBand, CorrelationModeMod::CorrelationMode correlationMode, vector<AtmPhaseCorrectionMod::AtmPhaseCorrection > atmPhaseCorrection);
	
	

	
	bool compareRequiredValue(vector<Tag>  antennaId, vector<Tag>  dataDescriptionId, vector<int>  feedId, Tag processorId, vector<Tag>  switchCycleId, int numAntenna, int numFeed, vector<int > numSubBand, CorrelationModeMod::CorrelationMode correlationMode, vector<AtmPhaseCorrectionMod::AtmPhaseCorrection > atmPhaseCorrection); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the ConfigDescriptionRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(ConfigDescriptionRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	ConfigDescriptionTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a ConfigDescriptionRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	ConfigDescriptionRow (ConfigDescriptionTable &table);

	/**
	 * Create a ConfigDescriptionRow using a copy constructor mechanism.
	 * <p>
	 * Given a ConfigDescriptionRow row and a ConfigDescriptionTable table, the method creates a new
	 * ConfigDescriptionRow owned by table. Each attribute of the created row is a copy (deep)
	 * of the corresponding attribute of row. The method does not add the created
	 * row to its table, its simply parents it to table, a call to the add method
	 * has to be done in order to get the row added (very likely after having modified
	 * some of its attributes).
	 * If row is null then the method returns a row with default values for its attributes. 
	 *
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 * @param row  The row which is to be copied.
	 */
	 ConfigDescriptionRow (ConfigDescriptionTable &table, ConfigDescriptionRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute numAntenna
	
	

	int numAntenna;

	
	
 	

	
	// ===> Attribute numFeed
	
	

	int numFeed;

	
	
 	

	
	// ===> Attribute numSubBand
	
	

	vector<int > numSubBand;

	
	
 	

	
	// ===> Attribute phasedArrayList, which is optional
	
	
	bool phasedArrayListExists;
	

	vector<int > phasedArrayList;

	
	
 	

	
	// ===> Attribute correlationMode
	
	

	CorrelationModeMod::CorrelationMode correlationMode;

	
	
 	

	
	// ===> Attribute flagAnt, which is optional
	
	
	bool flagAntExists;
	

	vector<bool > flagAnt;

	
	
 	

	
	// ===> Attribute configDescriptionId
	
	

	Tag configDescriptionId;

	
	
 	
 	/**
 	 * Set configDescriptionId with the specified Tag value.
 	 * @param configDescriptionId The Tag value to which configDescriptionId is to be set.
		
 		
			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setConfigDescriptionId (Tag configDescriptionId);
  		
	

	
	// ===> Attribute atmPhaseCorrection
	
	

	vector<AtmPhaseCorrectionMod::AtmPhaseCorrection > atmPhaseCorrection;

	
	
 	

	
	// ===> Attribute assocNature, which is optional
	
	
	bool assocNatureExists;
	

	vector<SpectralResolutionTypeMod::SpectralResolutionType > assocNature;

	
	
 	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId
	
	

	vector<Tag>  antennaId;

	
	
 	

	
	// ===> Attribute assocConfigDescriptionId, which is optional
	
	
	bool assocConfigDescriptionIdExists;
	

	vector<Tag>  assocConfigDescriptionId;

	
	
 	

	
	// ===> Attribute dataDescriptionId
	
	

	vector<Tag>  dataDescriptionId;

	
	
 	

	
	// ===> Attribute feedId
	
	

	vector<int>  feedId;

	
	
 	

	
	// ===> Attribute processorId
	
	

	Tag processorId;

	
	
 	

	
	// ===> Attribute switchCycleId
	
	

	vector<Tag>  switchCycleId;

	
	
 	

	///////////
	// Links //
	///////////
	
	
		

	 

	

	
		


	

	
		


	

	
		


	

	
		
	

	

	
		


	


};

} // End namespace asdm

#endif /* ConfigDescription_CLASS */