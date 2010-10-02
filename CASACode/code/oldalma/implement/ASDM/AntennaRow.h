
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
 * File AntennaRow.h
 */
 
#ifndef AntennaRow_CLASS
#define AntennaRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::AntennaRowIDL;
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




	

	

	
#include "CAntennaMake.h"
using namespace AntennaMakeMod;
	

	
#include "CAntennaType.h"
using namespace AntennaTypeMod;
	

	

	

	

	

	

	

	

	

	



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

/*\file Antenna.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::AntennaTable;


// class asdm::AntennaRow;
class AntennaRow;

// class asdm::StationRow;
class StationRow;
	

/**
 * The AntennaRow class is a row of a AntennaTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class AntennaRow {
friend class asdm::AntennaTable;

public:

	virtual ~AntennaRow();

	/**
	 * Return the table to which this row belongs.
	 */
	AntennaTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a AntennaRowIDL struct.
	 */
	AntennaRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct AntennaRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (AntennaRowIDL x) throw(ConversionException);
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
	
	
	// ===> Attribute antennaId
	
	
	

	
 	/**
 	 * Get antennaId.
 	 * @return antennaId as Tag
 	 */
 	Tag getAntennaId() const;
	
 
 	
 	
	
	


	
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
  		
	
	
	


	
	// ===> Attribute antennaMake
	
	
	

	
 	/**
 	 * Get antennaMake.
 	 * @return antennaMake as AntennaMakeMod::AntennaMake
 	 */
 	AntennaMakeMod::AntennaMake getAntennaMake() const;
	
 
 	
 	
 	/**
 	 * Set antennaMake with the specified AntennaMakeMod::AntennaMake.
 	 * @param antennaMake The AntennaMakeMod::AntennaMake value to which antennaMake is to be set.
 	 
 		
 			
 	 */
 	void setAntennaMake (AntennaMakeMod::AntennaMake antennaMake);
  		
	
	
	


	
	// ===> Attribute antennaType
	
	
	

	
 	/**
 	 * Get antennaType.
 	 * @return antennaType as AntennaTypeMod::AntennaType
 	 */
 	AntennaTypeMod::AntennaType getAntennaType() const;
	
 
 	
 	
 	/**
 	 * Set antennaType with the specified AntennaTypeMod::AntennaType.
 	 * @param antennaType The AntennaTypeMod::AntennaType value to which antennaType is to be set.
 	 
 		
 			
 	 */
 	void setAntennaType (AntennaTypeMod::AntennaType antennaType);
  		
	
	
	


	
	// ===> Attribute xPosition
	
	
	

	
 	/**
 	 * Get xPosition.
 	 * @return xPosition as Length
 	 */
 	Length getXPosition() const;
	
 
 	
 	
 	/**
 	 * Set xPosition with the specified Length.
 	 * @param xPosition The Length value to which xPosition is to be set.
 	 
 		
 			
 	 */
 	void setXPosition (Length xPosition);
  		
	
	
	


	
	// ===> Attribute yPosition
	
	
	

	
 	/**
 	 * Get yPosition.
 	 * @return yPosition as Length
 	 */
 	Length getYPosition() const;
	
 
 	
 	
 	/**
 	 * Set yPosition with the specified Length.
 	 * @param yPosition The Length value to which yPosition is to be set.
 	 
 		
 			
 	 */
 	void setYPosition (Length yPosition);
  		
	
	
	


	
	// ===> Attribute zPosition
	
	
	

	
 	/**
 	 * Get zPosition.
 	 * @return zPosition as Length
 	 */
 	Length getZPosition() const;
	
 
 	
 	
 	/**
 	 * Set zPosition with the specified Length.
 	 * @param zPosition The Length value to which zPosition is to be set.
 	 
 		
 			
 	 */
 	void setZPosition (Length zPosition);
  		
	
	
	


	
	// ===> Attribute time
	
	
	

	
 	/**
 	 * Get time.
 	 * @return time as ArrayTime
 	 */
 	ArrayTime getTime() const;
	
 
 	
 	
 	/**
 	 * Set time with the specified ArrayTime.
 	 * @param time The ArrayTime value to which time is to be set.
 	 
 		
 			
 	 */
 	void setTime (ArrayTime time);
  		
	
	
	


	
	// ===> Attribute xOffset
	
	
	

	
 	/**
 	 * Get xOffset.
 	 * @return xOffset as Length
 	 */
 	Length getXOffset() const;
	
 
 	
 	
 	/**
 	 * Set xOffset with the specified Length.
 	 * @param xOffset The Length value to which xOffset is to be set.
 	 
 		
 			
 	 */
 	void setXOffset (Length xOffset);
  		
	
	
	


	
	// ===> Attribute yOffset
	
	
	

	
 	/**
 	 * Get yOffset.
 	 * @return yOffset as Length
 	 */
 	Length getYOffset() const;
	
 
 	
 	
 	/**
 	 * Set yOffset with the specified Length.
 	 * @param yOffset The Length value to which yOffset is to be set.
 	 
 		
 			
 	 */
 	void setYOffset (Length yOffset);
  		
	
	
	


	
	// ===> Attribute zOffset
	
	
	

	
 	/**
 	 * Get zOffset.
 	 * @return zOffset as Length
 	 */
 	Length getZOffset() const;
	
 
 	
 	
 	/**
 	 * Set zOffset with the specified Length.
 	 * @param zOffset The Length value to which zOffset is to be set.
 	 
 		
 			
 	 */
 	void setZOffset (Length zOffset);
  		
	
	
	


	
	// ===> Attribute dishDiameter
	
	
	

	
 	/**
 	 * Get dishDiameter.
 	 * @return dishDiameter as Length
 	 */
 	Length getDishDiameter() const;
	
 
 	
 	
 	/**
 	 * Set dishDiameter with the specified Length.
 	 * @param dishDiameter The Length value to which dishDiameter is to be set.
 	 
 		
 			
 	 */
 	void setDishDiameter (Length dishDiameter);
  		
	
	
	


	
	// ===> Attribute flagRow
	
	
	

	
 	/**
 	 * Get flagRow.
 	 * @return flagRow as bool
 	 */
 	bool getFlagRow() const;
	
 
 	
 	
 	/**
 	 * Set flagRow with the specified bool.
 	 * @param flagRow The bool value to which flagRow is to be set.
 	 
 		
 			
 	 */
 	void setFlagRow (bool flagRow);
  		
	
	
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute assocAntennaId, which is optional
	
	
	
	/**
	 * The attribute assocAntennaId is optional. Return true if this attribute exists.
	 * @return true if and only if the assocAntennaId attribute exists. 
	 */
	bool isAssocAntennaIdExists() const;
	

	
 	/**
 	 * Get assocAntennaId, which is optional.
 	 * @return assocAntennaId as Tag
 	 * @throws IllegalAccessException If assocAntennaId does not exist.
 	 */
 	Tag getAssocAntennaId() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set assocAntennaId with the specified Tag.
 	 * @param assocAntennaId The Tag value to which assocAntennaId is to be set.
 	 
 		
 	 */
 	void setAssocAntennaId (Tag assocAntennaId);
		
	
	
	
	/**
	 * Mark assocAntennaId, which is an optional field, as non-existent.
	 */
	void clearAssocAntennaId ();
	


	
	// ===> Attribute stationId
	
	
	

	
 	/**
 	 * Get stationId.
 	 * @return stationId as Tag
 	 */
 	Tag getStationId() const;
	
 
 	
 	
 	/**
 	 * Set stationId with the specified Tag.
 	 * @param stationId The Tag value to which stationId is to be set.
 	 
 		
 			
 	 */
 	void setStationId (Tag stationId);
  		
	
	
	


	///////////
	// Links //
	///////////
	
	

	
		
		
			
	// ===> Optional link from a row of Antenna table to a row of Antenna table.

	/**
	 * The link to table Antenna is optional. Return true if this link exists.
	 * @return true if and only if the Antenna link exists. 
	 */
	bool isAssociatedAntennaExists() const;

	/**
	 * Get the optional row in table Antenna by traversing the defined link to that table.
	 * @return A row in Antenna table.
	 * @throws NoSuchRow if there is no such row in table Antenna or the link does not exist.
	 */
	AntennaRow *getAssociatedAntenna() const throw(NoSuchRow);
	
	/**
	 * Set the values of the link attributes needed to link this row to a row in table Antenna.
	 */
	void setAssociatedAntennaLink(Tag assocAntennaId);


		
		
	

	

	
		
	/**
	 * stationId pointer to the row in the Station table having Station.stationId == stationId
	 * @return a StationRow*
	 * 
	 
	 */
	 StationRow* getStationUsingStationId();
	 

	

	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this AntennaRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(Tag stationId, string name, AntennaMakeMod::AntennaMake antennaMake, AntennaTypeMod::AntennaType antennaType, Length xPosition, Length yPosition, Length zPosition, ArrayTime time, Length xOffset, Length yOffset, Length zOffset, Length dishDiameter, bool flagRow);
	
	

	
	bool compareRequiredValue(Tag stationId, string name, AntennaMakeMod::AntennaMake antennaMake, AntennaTypeMod::AntennaType antennaType, Length xPosition, Length yPosition, Length zPosition, ArrayTime time, Length xOffset, Length yOffset, Length zOffset, Length dishDiameter, bool flagRow); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the AntennaRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(AntennaRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	AntennaTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a AntennaRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	AntennaRow (AntennaTable &table);

	/**
	 * Create a AntennaRow using a copy constructor mechanism.
	 * <p>
	 * Given a AntennaRow row and a AntennaTable table, the method creates a new
	 * AntennaRow owned by table. Each attribute of the created row is a copy (deep)
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
	 AntennaRow (AntennaTable &table, AntennaRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId
	
	

	Tag antennaId;

	
	
 	
 	/**
 	 * Set antennaId with the specified Tag value.
 	 * @param antennaId The Tag value to which antennaId is to be set.
		
 		
			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setAntennaId (Tag antennaId);
  		
	

	
	// ===> Attribute name
	
	

	string name;

	
	
 	

	
	// ===> Attribute antennaMake
	
	

	AntennaMakeMod::AntennaMake antennaMake;

	
	
 	

	
	// ===> Attribute antennaType
	
	

	AntennaTypeMod::AntennaType antennaType;

	
	
 	

	
	// ===> Attribute xPosition
	
	

	Length xPosition;

	
	
 	

	
	// ===> Attribute yPosition
	
	

	Length yPosition;

	
	
 	

	
	// ===> Attribute zPosition
	
	

	Length zPosition;

	
	
 	

	
	// ===> Attribute time
	
	

	ArrayTime time;

	
	
 	

	
	// ===> Attribute xOffset
	
	

	Length xOffset;

	
	
 	

	
	// ===> Attribute yOffset
	
	

	Length yOffset;

	
	
 	

	
	// ===> Attribute zOffset
	
	

	Length zOffset;

	
	
 	

	
	// ===> Attribute dishDiameter
	
	

	Length dishDiameter;

	
	
 	

	
	// ===> Attribute flagRow
	
	

	bool flagRow;

	
	
 	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute assocAntennaId, which is optional
	
	
	bool assocAntennaIdExists;
	

	Tag assocAntennaId;

	
	
 	

	
	// ===> Attribute stationId
	
	

	Tag stationId;

	
	
 	

	///////////
	// Links //
	///////////
	
		
		
	

	
		

	 

	


};

} // End namespace asdm

#endif /* Antenna_CLASS */
