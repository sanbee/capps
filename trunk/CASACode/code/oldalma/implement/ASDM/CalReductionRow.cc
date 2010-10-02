
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
 * File CalReductionRow.cpp
 */
 
#include <vector>
using std::vector;

#include <set>
using std::set;

#include <ASDM.h>
#include <CalReductionRow.h>
#include <CalReductionTable.h>
	

using asdm::ASDM;
using asdm::CalReductionRow;
using asdm::CalReductionTable;


#include <Parser.h>
using asdm::Parser;

#include <EnumerationParser.h>
 
#include <InvalidArgumentException.h>
using asdm::InvalidArgumentException;

namespace asdm {

	CalReductionRow::~CalReductionRow() {
	}

	/**
	 * Return the table to which this row belongs.
	 */
	CalReductionTable &CalReductionRow::getTable() const {
		return table;
	}
	
	void CalReductionRow::isAdded() {
		hasBeenAdded = true;
	}
	
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a CalReductionRowIDL struct.
	 */
	CalReductionRowIDL *CalReductionRow::toIDL() const {
		CalReductionRowIDL *x = new CalReductionRowIDL ();
		
		// Fill the IDL structure.
	
		
	
  		
		
		
			
		x->calReductionId = calReductionId.toIDLTag();
			
		
	

	
  		
		
		
			
				
		x->numApplied = numApplied;
 				
 			
		
	

	
  		
		
		
			
				
		x->numParam = numParam;
 				
 			
		
	

	
  		
		
		
			
		x->timeReduced = timeReduced.toIDLArrayTime();
			
		
	

	
  		
		
		
			
		x->calAppliedArray.length(calAppliedArray.size());
		for (unsigned int i = 0; i < calAppliedArray.size(); ++i) {
			
				
			x->calAppliedArray[i] = CORBA::string_dup(calAppliedArray.at(i).c_str());
				
	 		
	 	}
			
		
	

	
  		
		
		
			
		x->paramSet.length(paramSet.size());
		for (unsigned int i = 0; i < paramSet.size(); ++i) {
			
				
			x->paramSet[i] = CORBA::string_dup(paramSet.at(i).c_str());
				
	 		
	 	}
			
		
	

	
  		
		
		
			
				
		x->messages = CORBA::string_dup(messages.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->software = CORBA::string_dup(software.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->softwareVersion = CORBA::string_dup(softwareVersion.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->numInvalidConditions = numInvalidConditions;
 				
 			
		
	

	
  		
		
		
			
		x->invalidConditions.length(invalidConditions.size());
		for (unsigned int i = 0; i < invalidConditions.size(); ++i) {
			
				
			x->invalidConditions[i] = invalidConditions.at(i);
	 			
	 		
	 	}
			
		
	

	
	
		
		
		return x;
	
	}
#endif
	

#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct CalReductionRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void CalReductionRow::setFromIDL (CalReductionRowIDL x) throw(ConversionException) {
		try {
		// Fill the values from x.
	
		
	
		
		
			
		setCalReductionId(Tag (x.calReductionId));
			
 		
		
	

	
		
		
			
		setNumApplied(x.numApplied);
  			
 		
		
	

	
		
		
			
		setNumParam(x.numParam);
  			
 		
		
	

	
		
		
			
		setTimeReduced(ArrayTime (x.timeReduced));
			
 		
		
	

	
		
		
			
		calAppliedArray .clear();
		for (unsigned int i = 0; i <x.calAppliedArray.length(); ++i) {
			
			calAppliedArray.push_back(string (x.calAppliedArray[i]));
			
		}
			
  		
		
	

	
		
		
			
		paramSet .clear();
		for (unsigned int i = 0; i <x.paramSet.length(); ++i) {
			
			paramSet.push_back(string (x.paramSet[i]));
			
		}
			
  		
		
	

	
		
		
			
		setMessages(string (x.messages));
			
 		
		
	

	
		
		
			
		setSoftware(string (x.software));
			
 		
		
	

	
		
		
			
		setSoftwareVersion(string (x.softwareVersion));
			
 		
		
	

	
		
		
			
		setNumInvalidConditions(x.numInvalidConditions);
  			
 		
		
	

	
		
		
			
		invalidConditions .clear();
		for (unsigned int i = 0; i <x.invalidConditions.length(); ++i) {
			
			invalidConditions.push_back(x.invalidConditions[i]);
  			
		}
			
  		
		
	

	
	
		
		} catch (IllegalAccessException err) {
			throw new ConversionException (err.getMessage(),"CalReduction");
		}
	}
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string CalReductionRow::toXML() const {
		string buf;
		buf.append("<row> \n");
		
	
		
  	
 		
		
		Parser::toXML(calReductionId, "calReductionId", buf);
		
		
	

  	
 		
		
		Parser::toXML(numApplied, "numApplied", buf);
		
		
	

  	
 		
		
		Parser::toXML(numParam, "numParam", buf);
		
		
	

  	
 		
		
		Parser::toXML(timeReduced, "timeReduced", buf);
		
		
	

  	
 		
		
		Parser::toXML(calAppliedArray, "calAppliedArray", buf);
		
		
	

  	
 		
		
		Parser::toXML(paramSet, "paramSet", buf);
		
		
	

  	
 		
		
		Parser::toXML(messages, "messages", buf);
		
		
	

  	
 		
		
		Parser::toXML(software, "software", buf);
		
		
	

  	
 		
		
		Parser::toXML(softwareVersion, "softwareVersion", buf);
		
		
	

  	
 		
		
		Parser::toXML(numInvalidConditions, "numInvalidConditions", buf);
		
		
	

  	
 		
		
			buf.append(EnumerationParser::toXML("invalidConditions", invalidConditions));
		
		
	

	
	
		
		
		buf.append("</row>\n");
		return buf;
	}

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void CalReductionRow::setFromXML (string rowDoc) throw(ConversionException) {
		Parser row(rowDoc);
		string s = "";
		try {
	
		
	
  		
			
	  	setCalReductionId(Parser::getTag("calReductionId","CalReduction",rowDoc));
			
		
	

	
  		
			
	  	setNumApplied(Parser::getInteger("numApplied","CalReduction",rowDoc));
			
		
	

	
  		
			
	  	setNumParam(Parser::getInteger("numParam","CalReduction",rowDoc));
			
		
	

	
  		
			
	  	setTimeReduced(Parser::getArrayTime("timeReduced","CalReduction",rowDoc));
			
		
	

	
  		
			
					
	  	setCalAppliedArray(Parser::get1DString("calAppliedArray","CalReduction",rowDoc));
	  			
	  		
		
	

	
  		
			
					
	  	setParamSet(Parser::get1DString("paramSet","CalReduction",rowDoc));
	  			
	  		
		
	

	
  		
			
	  	setMessages(Parser::getString("messages","CalReduction",rowDoc));
			
		
	

	
  		
			
	  	setSoftware(Parser::getString("software","CalReduction",rowDoc));
			
		
	

	
  		
			
	  	setSoftwareVersion(Parser::getString("softwareVersion","CalReduction",rowDoc));
			
		
	

	
  		
			
	  	setNumInvalidConditions(Parser::getInteger("numInvalidConditions","CalReduction",rowDoc));
			
		
	

	
		
		
		
		invalidConditions = EnumerationParser::getInvalidatingCondition1D("invalidConditions","CalReduction",rowDoc);			
		
		
		
	

	
	
		
		} catch (IllegalAccessException err) {
			throw ConversionException (err.getMessage(),"CalReduction");
		}
	}
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	

	
 	/**
 	 * Get calReductionId.
 	 * @return calReductionId as Tag
 	 */
 	Tag CalReductionRow::getCalReductionId() const {
	
  		return calReductionId;
 	}

 	/**
 	 * Set calReductionId with the specified Tag.
 	 * @param calReductionId The Tag value to which calReductionId is to be set.
 	 
 	
 		
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 	
 	 */
 	void CalReductionRow::setCalReductionId (Tag calReductionId)  {
  	
  	
  		if (hasBeenAdded) {
 		
			throw IllegalAccessException("calReductionId", "CalReduction");
		
  		}
  	
 		this->calReductionId = calReductionId;
	
 	}
	
	

	

	
 	/**
 	 * Get numApplied.
 	 * @return numApplied as int
 	 */
 	int CalReductionRow::getNumApplied() const {
	
  		return numApplied;
 	}

 	/**
 	 * Set numApplied with the specified int.
 	 * @param numApplied The int value to which numApplied is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setNumApplied (int numApplied)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->numApplied = numApplied;
	
 	}
	
	

	

	
 	/**
 	 * Get numParam.
 	 * @return numParam as int
 	 */
 	int CalReductionRow::getNumParam() const {
	
  		return numParam;
 	}

 	/**
 	 * Set numParam with the specified int.
 	 * @param numParam The int value to which numParam is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setNumParam (int numParam)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->numParam = numParam;
	
 	}
	
	

	

	
 	/**
 	 * Get timeReduced.
 	 * @return timeReduced as ArrayTime
 	 */
 	ArrayTime CalReductionRow::getTimeReduced() const {
	
  		return timeReduced;
 	}

 	/**
 	 * Set timeReduced with the specified ArrayTime.
 	 * @param timeReduced The ArrayTime value to which timeReduced is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setTimeReduced (ArrayTime timeReduced)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->timeReduced = timeReduced;
	
 	}
	
	

	

	
 	/**
 	 * Get calAppliedArray.
 	 * @return calAppliedArray as vector<string >
 	 */
 	vector<string > CalReductionRow::getCalAppliedArray() const {
	
  		return calAppliedArray;
 	}

 	/**
 	 * Set calAppliedArray with the specified vector<string >.
 	 * @param calAppliedArray The vector<string > value to which calAppliedArray is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setCalAppliedArray (vector<string > calAppliedArray)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->calAppliedArray = calAppliedArray;
	
 	}
	
	

	

	
 	/**
 	 * Get paramSet.
 	 * @return paramSet as vector<string >
 	 */
 	vector<string > CalReductionRow::getParamSet() const {
	
  		return paramSet;
 	}

 	/**
 	 * Set paramSet with the specified vector<string >.
 	 * @param paramSet The vector<string > value to which paramSet is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setParamSet (vector<string > paramSet)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->paramSet = paramSet;
	
 	}
	
	

	

	
 	/**
 	 * Get messages.
 	 * @return messages as string
 	 */
 	string CalReductionRow::getMessages() const {
	
  		return messages;
 	}

 	/**
 	 * Set messages with the specified string.
 	 * @param messages The string value to which messages is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setMessages (string messages)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->messages = messages;
	
 	}
	
	

	

	
 	/**
 	 * Get software.
 	 * @return software as string
 	 */
 	string CalReductionRow::getSoftware() const {
	
  		return software;
 	}

 	/**
 	 * Set software with the specified string.
 	 * @param software The string value to which software is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setSoftware (string software)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->software = software;
	
 	}
	
	

	

	
 	/**
 	 * Get softwareVersion.
 	 * @return softwareVersion as string
 	 */
 	string CalReductionRow::getSoftwareVersion() const {
	
  		return softwareVersion;
 	}

 	/**
 	 * Set softwareVersion with the specified string.
 	 * @param softwareVersion The string value to which softwareVersion is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setSoftwareVersion (string softwareVersion)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->softwareVersion = softwareVersion;
	
 	}
	
	

	

	
 	/**
 	 * Get numInvalidConditions.
 	 * @return numInvalidConditions as int
 	 */
 	int CalReductionRow::getNumInvalidConditions() const {
	
  		return numInvalidConditions;
 	}

 	/**
 	 * Set numInvalidConditions with the specified int.
 	 * @param numInvalidConditions The int value to which numInvalidConditions is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setNumInvalidConditions (int numInvalidConditions)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->numInvalidConditions = numInvalidConditions;
	
 	}
	
	

	

	
 	/**
 	 * Get invalidConditions.
 	 * @return invalidConditions as vector<InvalidatingConditionMod::InvalidatingCondition >
 	 */
 	vector<InvalidatingConditionMod::InvalidatingCondition > CalReductionRow::getInvalidConditions() const {
	
  		return invalidConditions;
 	}

 	/**
 	 * Set invalidConditions with the specified vector<InvalidatingConditionMod::InvalidatingCondition >.
 	 * @param invalidConditions The vector<InvalidatingConditionMod::InvalidatingCondition > value to which invalidConditions is to be set.
 	 
 	
 		
 	 */
 	void CalReductionRow::setInvalidConditions (vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->invalidConditions = invalidConditions;
	
 	}
	
	

	
	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	
	
	/**
	 * Create a CalReductionRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	CalReductionRow::CalReductionRow (CalReductionTable &t) : table(t) {
		hasBeenAdded = false;
		
	
	

	

	

	

	

	

	

	

	

	

	

	
	
	
	
	

	

	

	

	

	

	

	

	

	

	
	
	}
	
	CalReductionRow::CalReductionRow (CalReductionTable &t, CalReductionRow &row) : table(t) {
		hasBeenAdded = false;
		
		if (&row == 0) {
	
	
	

	

	

	

	

	

	

	

	

	

	

			
		}
		else {
	
		
			calReductionId = row.calReductionId;
		
		
		
		
			numApplied = row.numApplied;
		
			numParam = row.numParam;
		
			timeReduced = row.timeReduced;
		
			calAppliedArray = row.calAppliedArray;
		
			paramSet = row.paramSet;
		
			messages = row.messages;
		
			software = row.software;
		
			softwareVersion = row.softwareVersion;
		
			numInvalidConditions = row.numInvalidConditions;
		
			invalidConditions = row.invalidConditions;
		
		
		
		
		}	
	}

	
	bool CalReductionRow::compareNoAutoInc(int numApplied, int numParam, ArrayTime timeReduced, vector<string > calAppliedArray, vector<string > paramSet, string messages, string software, string softwareVersion, int numInvalidConditions, vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions) {
		bool result;
		result = true;
		
	
		
		result = result && (this->numApplied == numApplied);
		
		if (!result) return false;
	

	
		
		result = result && (this->numParam == numParam);
		
		if (!result) return false;
	

	
		
		result = result && (this->timeReduced == timeReduced);
		
		if (!result) return false;
	

	
		
		result = result && (this->calAppliedArray == calAppliedArray);
		
		if (!result) return false;
	

	
		
		result = result && (this->paramSet == paramSet);
		
		if (!result) return false;
	

	
		
		result = result && (this->messages == messages);
		
		if (!result) return false;
	

	
		
		result = result && (this->software == software);
		
		if (!result) return false;
	

	
		
		result = result && (this->softwareVersion == softwareVersion);
		
		if (!result) return false;
	

	
		
		result = result && (this->numInvalidConditions == numInvalidConditions);
		
		if (!result) return false;
	

	
		
		result = result && (this->invalidConditions == invalidConditions);
		
		if (!result) return false;
	

		return result;
	}	
	
	
	
	bool CalReductionRow::compareRequiredValue(int numApplied, int numParam, ArrayTime timeReduced, vector<string > calAppliedArray, vector<string > paramSet, string messages, string software, string softwareVersion, int numInvalidConditions, vector<InvalidatingConditionMod::InvalidatingCondition > invalidConditions) {
		bool result;
		result = true;
		
	
		if (!(this->numApplied == numApplied)) return false;
	

	
		if (!(this->numParam == numParam)) return false;
	

	
		if (!(this->timeReduced == timeReduced)) return false;
	

	
		if (!(this->calAppliedArray == calAppliedArray)) return false;
	

	
		if (!(this->paramSet == paramSet)) return false;
	

	
		if (!(this->messages == messages)) return false;
	

	
		if (!(this->software == software)) return false;
	

	
		if (!(this->softwareVersion == softwareVersion)) return false;
	

	
		if (!(this->numInvalidConditions == numInvalidConditions)) return false;
	

	
		if (!(this->invalidConditions == invalidConditions)) return false;
	

		return result;
	}
	
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the CalReductionRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool CalReductionRow::equalByRequiredValue(CalReductionRow* x) {
		
			
		if (this->numApplied != x->numApplied) return false;
			
		if (this->numParam != x->numParam) return false;
			
		if (this->timeReduced != x->timeReduced) return false;
			
		if (this->calAppliedArray != x->calAppliedArray) return false;
			
		if (this->paramSet != x->paramSet) return false;
			
		if (this->messages != x->messages) return false;
			
		if (this->software != x->software) return false;
			
		if (this->softwareVersion != x->softwareVersion) return false;
			
		if (this->numInvalidConditions != x->numInvalidConditions) return false;
			
		if (this->invalidConditions != x->invalidConditions) return false;
			
		
		return true;
	}	
	

} // End namespace asdm
 
