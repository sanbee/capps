
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
 * File ProcessorRow.cpp
 */
 
#include <vector>
using std::vector;

#include <set>
using std::set;

#include <ASDM.h>
#include <ProcessorRow.h>
#include <ProcessorTable.h>
	

using asdm::ASDM;
using asdm::ProcessorRow;
using asdm::ProcessorTable;


#include <Parser.h>
using asdm::Parser;

#include <EnumerationParser.h>
 
#include <InvalidArgumentException.h>
using asdm::InvalidArgumentException;

namespace asdm {
	ProcessorRow::~ProcessorRow() {
	}

	/**
	 * Return the table to which this row belongs.
	 */
	ProcessorTable &ProcessorRow::getTable() const {
		return table;
	}

	bool ProcessorRow::isAdded() const {
		return hasBeenAdded;
	}	

	void ProcessorRow::isAdded(bool added) {
		hasBeenAdded = added;
	}
	
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a ProcessorRowIDL struct.
	 */
	ProcessorRowIDL *ProcessorRow::toIDL() const {
		ProcessorRowIDL *x = new ProcessorRowIDL ();
		
		// Fill the IDL structure.
	
		
	
  		
		
		
			
		x->processorId = processorId.toIDLTag();
			
		
	

	
  		
		
		
			
		x->modeId = modeId.toIDLTag();
			
		
	

	
  		
		
		
			
				
		x->processorType = processorType;
 				
 			
		
	

	
  		
		
		
			
				
		x->processorSubType = processorSubType;
 				
 			
		
	

	
	
		
		
		return x;
	
	}
#endif
	

#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct ProcessorRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void ProcessorRow::setFromIDL (ProcessorRowIDL x){
		try {
		// Fill the values from x.
	
		
	
		
		
			
		setProcessorId(Tag (x.processorId));
			
 		
		
	

	
		
		
			
		setModeId(Tag (x.modeId));
			
 		
		
	

	
		
		
			
		setProcessorType(x.processorType);
  			
 		
		
	

	
		
		
			
		setProcessorSubType(x.processorSubType);
  			
 		
		
	

	
	
		
		} catch (IllegalAccessException err) {
			throw ConversionException (err.getMessage(),"Processor");
		}
	}
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string ProcessorRow::toXML() const {
		string buf;
		buf.append("<row> \n");
		
	
		
  	
 		
		
		Parser::toXML(processorId, "processorId", buf);
		
		
	

  	
 		
		
		Parser::toXML(modeId, "modeId", buf);
		
		
	

  	
 		
		
			buf.append(EnumerationParser::toXML("processorType", processorType));
		
		
	

  	
 		
		
			buf.append(EnumerationParser::toXML("processorSubType", processorSubType));
		
		
	

	
	
		
		
		buf.append("</row>\n");
		return buf;
	}

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void ProcessorRow::setFromXML (string rowDoc) {
		Parser row(rowDoc);
		string s = "";
		try {
	
		
	
  		
			
	  	setProcessorId(Parser::getTag("processorId","Processor",rowDoc));
			
		
	

	
  		
			
	  	setModeId(Parser::getTag("modeId","Processor",rowDoc));
			
		
	

	
		
		
		
		processorType = EnumerationParser::getProcessorType("processorType","Processor",rowDoc);
		
		
		
	

	
		
		
		
		processorSubType = EnumerationParser::getProcessorSubType("processorSubType","Processor",rowDoc);
		
		
		
	

	
	
		
		} catch (IllegalAccessException err) {
			throw ConversionException (err.getMessage(),"Processor");
		}
	}
	
	void ProcessorRow::toBin(EndianOSStream& eoss) {
	
	
	
	
		
	processorId.toBin(eoss);
		
	

	
	
		
	modeId.toBin(eoss);
		
	

	
	
		
					
			eoss.writeInt(processorType);
				
		
	

	
	
		
					
			eoss.writeInt(processorSubType);
				
		
	


	
	
	}
	
void ProcessorRow::processorIdFromBin(EndianISStream& eiss) {
		
	
		
		
		processorId =  Tag::fromBin(eiss);
		
	
	
}
void ProcessorRow::modeIdFromBin(EndianISStream& eiss) {
		
	
		
		
		modeId =  Tag::fromBin(eiss);
		
	
	
}
void ProcessorRow::processorTypeFromBin(EndianISStream& eiss) {
		
	
	
		
			
		processorType = CProcessorType::from_int(eiss.readInt());
			
		
	
	
}
void ProcessorRow::processorSubTypeFromBin(EndianISStream& eiss) {
		
	
	
		
			
		processorSubType = CProcessorSubType::from_int(eiss.readInt());
			
		
	
	
}

		
	
	ProcessorRow* ProcessorRow::fromBin(EndianISStream& eiss, ProcessorTable& table, const vector<string>& attributesSeq) {
		ProcessorRow* row = new  ProcessorRow(table);
		
		map<string, ProcessorAttributeFromBin>::iterator iter ;
		for (unsigned int i = 0; i < attributesSeq.size(); i++) {
			iter = row->fromBinMethods.find(attributesSeq.at(i));
			if (iter == row->fromBinMethods.end()) {
				throw ConversionException("There is not method to read an attribute '"+attributesSeq.at(i)+"'.", "ProcessorTable");
			}
			(row->*(row->fromBinMethods[ attributesSeq.at(i) ] ))(eiss);
		}				
		return row;
	}
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	

	
 	/**
 	 * Get processorId.
 	 * @return processorId as Tag
 	 */
 	Tag ProcessorRow::getProcessorId() const {
	
  		return processorId;
 	}

 	/**
 	 * Set processorId with the specified Tag.
 	 * @param processorId The Tag value to which processorId is to be set.
 	 
 	
 		
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 	
 	 */
 	void ProcessorRow::setProcessorId (Tag processorId)  {
  	
  	
  		if (hasBeenAdded) {
 		
			throw IllegalAccessException("processorId", "Processor");
		
  		}
  	
 		this->processorId = processorId;
	
 	}
	
	

	

	
 	/**
 	 * Get modeId.
 	 * @return modeId as Tag
 	 */
 	Tag ProcessorRow::getModeId() const {
	
  		return modeId;
 	}

 	/**
 	 * Set modeId with the specified Tag.
 	 * @param modeId The Tag value to which modeId is to be set.
 	 
 	
 		
 	 */
 	void ProcessorRow::setModeId (Tag modeId)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->modeId = modeId;
	
 	}
	
	

	

	
 	/**
 	 * Get processorType.
 	 * @return processorType as ProcessorTypeMod::ProcessorType
 	 */
 	ProcessorTypeMod::ProcessorType ProcessorRow::getProcessorType() const {
	
  		return processorType;
 	}

 	/**
 	 * Set processorType with the specified ProcessorTypeMod::ProcessorType.
 	 * @param processorType The ProcessorTypeMod::ProcessorType value to which processorType is to be set.
 	 
 	
 		
 	 */
 	void ProcessorRow::setProcessorType (ProcessorTypeMod::ProcessorType processorType)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->processorType = processorType;
	
 	}
	
	

	

	
 	/**
 	 * Get processorSubType.
 	 * @return processorSubType as ProcessorSubTypeMod::ProcessorSubType
 	 */
 	ProcessorSubTypeMod::ProcessorSubType ProcessorRow::getProcessorSubType() const {
	
  		return processorSubType;
 	}

 	/**
 	 * Set processorSubType with the specified ProcessorSubTypeMod::ProcessorSubType.
 	 * @param processorSubType The ProcessorSubTypeMod::ProcessorSubType value to which processorSubType is to be set.
 	 
 	
 		
 	 */
 	void ProcessorRow::setProcessorSubType (ProcessorSubTypeMod::ProcessorSubType processorSubType)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->processorSubType = processorSubType;
	
 	}
	
	

	
	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	
	
	/**
	 * Create a ProcessorRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	ProcessorRow::ProcessorRow (ProcessorTable &t) : table(t) {
		hasBeenAdded = false;
		
	
	

	

	

	

	
	
	
	
	

	

	
// This attribute is scalar and has an enumeration type. Let's initialize it to some valid value (the 1st of the enumeration).		
processorType = CProcessorType::from_int(0);
	

	
// This attribute is scalar and has an enumeration type. Let's initialize it to some valid value (the 1st of the enumeration).		
processorSubType = CProcessorSubType::from_int(0);
	

	
	
	 fromBinMethods["processorId"] = &ProcessorRow::processorIdFromBin; 
	 fromBinMethods["modeId"] = &ProcessorRow::modeIdFromBin; 
	 fromBinMethods["processorType"] = &ProcessorRow::processorTypeFromBin; 
	 fromBinMethods["processorSubType"] = &ProcessorRow::processorSubTypeFromBin; 
		
	
	
	}
	
	ProcessorRow::ProcessorRow (ProcessorTable &t, ProcessorRow &row) : table(t) {
		hasBeenAdded = false;
		
		if (&row == 0) {
	
	
	

	

	

	

			
		}
		else {
	
		
			processorId = row.processorId;
		
		
		
		
			modeId = row.modeId;
		
			processorType = row.processorType;
		
			processorSubType = row.processorSubType;
		
		
		
		
		}
		
		 fromBinMethods["processorId"] = &ProcessorRow::processorIdFromBin; 
		 fromBinMethods["modeId"] = &ProcessorRow::modeIdFromBin; 
		 fromBinMethods["processorType"] = &ProcessorRow::processorTypeFromBin; 
		 fromBinMethods["processorSubType"] = &ProcessorRow::processorSubTypeFromBin; 
			
	
			
	}

	
	bool ProcessorRow::compareNoAutoInc(Tag modeId, ProcessorTypeMod::ProcessorType processorType, ProcessorSubTypeMod::ProcessorSubType processorSubType) {
		bool result;
		result = true;
		
	
		
		result = result && (this->modeId == modeId);
		
		if (!result) return false;
	

	
		
		result = result && (this->processorType == processorType);
		
		if (!result) return false;
	

	
		
		result = result && (this->processorSubType == processorSubType);
		
		if (!result) return false;
	

		return result;
	}	
	
	
	
	bool ProcessorRow::compareRequiredValue(Tag modeId, ProcessorTypeMod::ProcessorType processorType, ProcessorSubTypeMod::ProcessorSubType processorSubType) {
		bool result;
		result = true;
		
	
		if (!(this->modeId == modeId)) return false;
	

	
		if (!(this->processorType == processorType)) return false;
	

	
		if (!(this->processorSubType == processorSubType)) return false;
	

		return result;
	}
	
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the ProcessorRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool ProcessorRow::equalByRequiredValue(ProcessorRow* x) {
		
			
		if (this->modeId != x->modeId) return false;
			
		if (this->processorType != x->processorType) return false;
			
		if (this->processorSubType != x->processorSubType) return false;
			
		
		return true;
	}	
	
/*
	 map<string, ProcessorAttributeFromBin> ProcessorRow::initFromBinMethods() {
		map<string, ProcessorAttributeFromBin> result;
		
		result["processorId"] = &ProcessorRow::processorIdFromBin;
		result["modeId"] = &ProcessorRow::modeIdFromBin;
		result["processorType"] = &ProcessorRow::processorTypeFromBin;
		result["processorSubType"] = &ProcessorRow::processorSubTypeFromBin;
		
		
			
		
		return result;	
	}
*/	
} // End namespace asdm
 
