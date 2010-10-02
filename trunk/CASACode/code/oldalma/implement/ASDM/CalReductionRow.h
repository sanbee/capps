
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
 * File CalReductionRow.h
 */
 
#ifndef CalReductionRow_CLASS
#define CalReductionRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::CalReductionRowIDL;
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




	

	

	

	

	

	

	

	

	

	

	
#include "CInvalidatingCondition.h"
using namespace InvalidatingConditionMod;
	



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

/*\file CalReduction.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::CalReductionTable;

	

/**
 * The CalReductionRow class is a row of a CalReductionTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class CalReductionRow {
friend class asdm::CalReductionTable;

public:

	virtual ~CalReductionRow();

	/**
	 * Return the table to which this row belongs.
	 */
	CalReductionTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a CalReductionRowIDL struct.
	 */
	CalReductionRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct CalReductionRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (CalReductionRowIDL x) throw(ConversionException);
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
	
	
	// ===> Attribute calReductionId
	
	
	

	
 	/**
 	 * Get calReductionId.
 	 * @return calReductionId as Tag
 	 */
 	Tag getCalReductionId() const;
	
 
 	
 	
	
	


	
	// ===> Attribute numApplied
	
	
	

	
 	/**
 	 * Get numApplied.
 	 * @return numApplied as int
 	 */
 	int getNumApplied() const;
	
 
 	
 	
 	/**
 	 * Set numApplied with the specified int.
 	 * @param numApplied The int value to which numApplied is to be set.
 	 
 		
 			
 	 */
 	void setNumApplied (int numApplied);
  		
	
	
	


	
	// ===> Attribute numParam
	
	
	

	
 	/**
 	 * Get numParam.
 	 * @return numParam as int
 	 */
 	int getNumParam() const;
	
 
 	
 	
 	/**
 	 * Set numParam with the specified int.
 	 * @param numParam The int value to which numParam is to be set.
 	 
 		
 			
 	 */
 	void setNumParam (int numParam);
  		
	
	
	


	
	// ===> Attribute timeReduced
	
	
	

	
 	/**
 	 * Get timeReduced.
 	 * @return timeReduced as ArrayTime
 	 */
 	ArrayTime getTimeReduced() const;
	
 
 	
 	
 	/**
 	 * Set timeReduced with the specified ArrayTime.
 	 * @param timeReduced The ArrayTime value to which timeReduced is to be set.
 	 
 		
 			
 	 */
 	void setTimeReduced (ArrayTime timeReduced);
  		
	
	
	


	
	// ===> Attribute calAppliedArray
	
	
	

	
 	/**
 	 * Get calAppliedArray.
 	 * @return calAppliedArray as vector<string >
 	 */
 	vector<string > getCalAppliedArray() const;
	
 
 	
 	
 	/**
 	 * Set calAppliedArray with the specified vector<string >.
 	 * @param calAppliedArray The vector<string > value to which calAppliedArray is to be set.
 	 
 		
 			
 	 */
 	void setCalAppliedArray (vector<string > calAppliedArray);
  		
	
	
	


	
	// ===> Attribute paramSet
	
	
	

	
 	/**
 	 * Get paramSet.
 	 * @return paramSet as vector<string >
 	 */
 	vector<string > getParamSet() const;
	
 
 	
 	
 	/**
 	 * Set paramSet with the specified vector<string >.
 	 * @param paramSet The vector<string > value to which paramSet is to be set.
 	 
 		
 			
 	 */
 	void setParamSet (vector<string > paramSet);
  		
	
	
	


	
	// ===> Attribute messages
	
	
	

	
 	/**
 	 * Get messages.
 	 * @return messages as string
 	 */
 	string getMessages() const;
	
 
 	
 	
 	/**
 	 * Set messages with the specified string.
 	 * @param messages The string value to which messages is to be set.
 	 
 		
 			
 	 */
 	void setMessages (string messages);
  		
	
	
	


	
	// ===> Attribute software
	
	
	

	
 	/**
 	 * Get software.
 	 * @return software as string
 	 */
 	string getSoftware() const;
	
 
 	
 	
 	/**
 	 * Set software with the specified string.
 	 * @param software The string value to which software is to be set.
 	 
 		
 			
 	 */
 	void setSoftware (string software);
  		
	
	
	


	
	// ===> Attribute softwareVersion
	
	
	

	
 	/**
 	 * Get softwareVersion.
 	 * @return softwareVersion as string
 	 */
 	string getSoftwareVersion() const;
	
 
 	
 	
 	/**
 	 * Set softwareVersion with the specified string.
 	 * @param softwareVersion The string value to which softwareVersion is to be set.
 	 
 		
 			
 	 */
 	void setSoftwareVersion (string softwareVersion);
  		
	
	
	


	
	// ===> Attribute numInvalidConditions
	
	
	

	
 	/**
 	 * Get numInvalidConditions.
 	 * @return numInvalidConditions as int
 	 */
 	int getNumInvalidConditions() const;
	
 
 	
 	
 	/**
 	 * Set numInvalidConditions with the specified int.
 	 * @param numInvalidConditions The int value to which numInvalidConditions is to be set.
 	 
 		
 			
 	 */
 	void setNumInvalidConditions (int numInvalidConditions);
  		
	
	
	


	
	// ===> Attribute invalidConditions
	
	
	

	
 	/**
 	 * Get invalidConditions.
 	 * @return invalidConditions as vector<InvalidatingConditionMod::InvalidatingCondition >
 	 */
 	vector<InvalidatingConditionMod::InvalidatingCondition > getInvalidConditions() const;
	
 
 	
 	
 	/**
 	 * Set invalidConditions with the specified vector<InvalidatingConditionMod::InvalidatingCondition >.
 	 * @param invalidConditions The vector<InvalidatingConditionMod::InvalidatingCondition > value to which invalidConditions is to be set.
 	 
 		
 			
 	 */
 	void setInvalidConditions (vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions);
  		
	
	
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	
	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this CalReductionRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(int numApplied, int numParam, ArrayTime timeReduced, vector<string > calAppliedArray, vector<string > paramSet, string messages, string software, string softwareVersion, int numInvalidConditions, vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions);
	
	

	
	bool compareRequiredValue(int numApplied, int numParam, ArrayTime timeReduced, vector<string > calAppliedArray, vector<string > paramSet, string messages, string software, string softwareVersion, int numInvalidConditions, vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the CalReductionRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(CalReductionRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	CalReductionTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a CalReductionRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	CalReductionRow (CalReductionTable &table);

	/**
	 * Create a CalReductionRow using a copy constructor mechanism.
	 * <p>
	 * Given a CalReductionRow row and a CalReductionTable table, the method creates a new
	 * CalReductionRow owned by table. Each attribute of the created row is a copy (deep)
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
	 CalReductionRow (CalReductionTable &table, CalReductionRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute calReductionId
	
	

	Tag calReductionId;

	
	
 	
 	/**
 	 * Set calReductionId with the specified Tag value.
 	 * @param calReductionId The Tag value to which calReductionId is to be set.
		
 		
			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setCalReductionId (Tag calReductionId);
  		
	

	
	// ===> Attribute numApplied
	
	

	int numApplied;

	
	
 	

	
	// ===> Attribute numParam
	
	

	int numParam;

	
	
 	

	
	// ===> Attribute timeReduced
	
	

	ArrayTime timeReduced;

	
	
 	

	
	// ===> Attribute calAppliedArray
	
	

	vector<string > calAppliedArray;

	
	
 	

	
	// ===> Attribute paramSet
	
	

	vector<string > paramSet;

	
	
 	

	
	// ===> Attribute messages
	
	

	string messages;

	
	
 	

	
	// ===> Attribute software
	
	

	string software;

	
	
 	

	
	// ===> Attribute softwareVersion
	
	

	string softwareVersion;

	
	
 	

	
	// ===> Attribute numInvalidConditions
	
	

	int numInvalidConditions;

	
	
 	

	
	// ===> Attribute invalidConditions
	
	

	vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions;

	
	
 	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	

};

} // End namespace asdm

#endif /* CalReduction_CLASS */
