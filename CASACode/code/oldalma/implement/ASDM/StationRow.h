
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
 * File StationRow.h
 */
 
#ifndef StationRow_CLASS
#define StationRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::StationRowIDL;
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




	

	

	
#include "CStationType.h"
using namespace StationTypeMod;
	

	



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

/*\file Station.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::StationTable;

	

/**
 * The StationRow class is a row of a StationTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class StationRow {
friend class asdm::StationTable;

public:

	virtual ~StationRow();

	/**
	 * Return the table to which this row belongs.
	 */
	StationTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a StationRowIDL struct.
	 */
	StationRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct StationRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (StationRowIDL x) throw(ConversionException);
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
	
	
	// ===> Attribute name
	
	
	

	
 	/**
 	 * Get name.
 	 * @return name as string
 	 */
 	string getName() const;
	
 
 	
 	
 	/**
 	 * Set name with the specified string.
 	 * @param name The string value to which name is to be set.
 	 
 		
 			
 	 */
 	void setName (string name);
  		
	
	
	


	
	// ===> Attribute position
	
	
	

	
 	/**
 	 * Get position.
 	 * @return position as vector<Length >
 	 */
 	vector<Length > getPosition() const;
	
 
 	
 	
 	/**
 	 * Set position with the specified vector<Length >.
 	 * @param position The vector<Length > value to which position is to be set.
 	 
 		
 			
 	 */
 	void setPosition (vector<Length > position);
  		
	
	
	


	
	// ===> Attribute type
	
	
	

	
 	/**
 	 * Get type.
 	 * @return type as StationTypeMod::StationType
 	 */
 	StationTypeMod::StationType getType() const;
	
 
 	
 	
 	/**
 	 * Set type with the specified StationTypeMod::StationType.
 	 * @param type The StationTypeMod::StationType value to which type is to be set.
 	 
 		
 			
 	 */
 	void setType (StationTypeMod::StationType type);
  		
	
	
	


	
	// ===> Attribute stationId
	
	
	

	
 	/**
 	 * Get stationId.
 	 * @return stationId as Tag
 	 */
 	Tag getStationId() const;
	
 
 	
 	
	
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	
	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this StationRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(string name, vector<Length > position, StationTypeMod::StationType type);
	
	

	
	bool compareRequiredValue(string name, vector<Length > position, StationTypeMod::StationType type); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the StationRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(StationRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	StationTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a StationRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	StationRow (StationTable &table);

	/**
	 * Create a StationRow using a copy constructor mechanism.
	 * <p>
	 * Given a StationRow row and a StationTable table, the method creates a new
	 * StationRow owned by table. Each attribute of the created row is a copy (deep)
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
	 StationRow (StationTable &table, StationRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute name
	
	

	string name;

	
	
 	

	
	// ===> Attribute position
	
	

	vector<Length > position;

	
	
 	

	
	// ===> Attribute type
	
	

	StationTypeMod::StationType type;

	
	
 	

	
	// ===> Attribute stationId
	
	

	Tag stationId;

	
	
 	
 	/**
 	 * Set stationId with the specified Tag value.
 	 * @param stationId The Tag value to which stationId is to be set.
		
 		
			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setStationId (Tag stationId);
  		
	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	

};

} // End namespace asdm

#endif /* Station_CLASS */
