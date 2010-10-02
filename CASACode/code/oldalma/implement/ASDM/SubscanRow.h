
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
 * File SubscanRow.h
 */
 
#ifndef SubscanRow_CLASS
#define SubscanRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::SubscanRowIDL;
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




	

	

	

	
#include "CSubscanIntent.h"
using namespace SubscanIntentMod;
	

	
#include "CSwitchingMode.h"
using namespace SwitchingModeMod;
	

	

	

	

	

	
#include "CCorrelatorCalibration.h"
using namespace CorrelatorCalibrationMod;
	



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

/*\file Subscan.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::SubscanTable;


// class asdm::ExecBlockRow;
class ExecBlockRow;
	

/**
 * The SubscanRow class is a row of a SubscanTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class SubscanRow {
friend class asdm::SubscanTable;

public:

	virtual ~SubscanRow();

	/**
	 * Return the table to which this row belongs.
	 */
	SubscanTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a SubscanRowIDL struct.
	 */
	SubscanRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct SubscanRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (SubscanRowIDL x) throw(ConversionException);
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
	
	
	// ===> Attribute subscanNumber
	
	
	

	
 	/**
 	 * Get subscanNumber.
 	 * @return subscanNumber as int
 	 */
 	int getSubscanNumber() const;
	
 
 	
 	
 	/**
 	 * Set subscanNumber with the specified int.
 	 * @param subscanNumber The int value to which subscanNumber is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setSubscanNumber (int subscanNumber);
  		
	
	
	


	
	// ===> Attribute startTime
	
	
	

	
 	/**
 	 * Get startTime.
 	 * @return startTime as ArrayTime
 	 */
 	ArrayTime getStartTime() const;
	
 
 	
 	
 	/**
 	 * Set startTime with the specified ArrayTime.
 	 * @param startTime The ArrayTime value to which startTime is to be set.
 	 
 		
 			
 	 */
 	void setStartTime (ArrayTime startTime);
  		
	
	
	


	
	// ===> Attribute endTime
	
	
	

	
 	/**
 	 * Get endTime.
 	 * @return endTime as ArrayTime
 	 */
 	ArrayTime getEndTime() const;
	
 
 	
 	
 	/**
 	 * Set endTime with the specified ArrayTime.
 	 * @param endTime The ArrayTime value to which endTime is to be set.
 	 
 		
 			
 	 */
 	void setEndTime (ArrayTime endTime);
  		
	
	
	


	
	// ===> Attribute subscanIntent
	
	
	

	
 	/**
 	 * Get subscanIntent.
 	 * @return subscanIntent as SubscanIntentMod::SubscanIntent
 	 */
 	SubscanIntentMod::SubscanIntent getSubscanIntent() const;
	
 
 	
 	
 	/**
 	 * Set subscanIntent with the specified SubscanIntentMod::SubscanIntent.
 	 * @param subscanIntent The SubscanIntentMod::SubscanIntent value to which subscanIntent is to be set.
 	 
 		
 			
 	 */
 	void setSubscanIntent (SubscanIntentMod::SubscanIntent subscanIntent);
  		
	
	
	


	
	// ===> Attribute subscanMode, which is optional
	
	
	
	/**
	 * The attribute subscanMode is optional. Return true if this attribute exists.
	 * @return true if and only if the subscanMode attribute exists. 
	 */
	bool isSubscanModeExists() const;
	

	
 	/**
 	 * Get subscanMode, which is optional.
 	 * @return subscanMode as SwitchingModeMod::SwitchingMode
 	 * @throws IllegalAccessException If subscanMode does not exist.
 	 */
 	SwitchingModeMod::SwitchingMode getSubscanMode() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set subscanMode with the specified SwitchingModeMod::SwitchingMode.
 	 * @param subscanMode The SwitchingModeMod::SwitchingMode value to which subscanMode is to be set.
 	 
 		
 	 */
 	void setSubscanMode (SwitchingModeMod::SwitchingMode subscanMode);
		
	
	
	
	/**
	 * Mark subscanMode, which is an optional field, as non-existent.
	 */
	void clearSubscanMode ();
	


	
	// ===> Attribute numberIntegration
	
	
	

	
 	/**
 	 * Get numberIntegration.
 	 * @return numberIntegration as int
 	 */
 	int getNumberIntegration() const;
	
 
 	
 	
 	/**
 	 * Set numberIntegration with the specified int.
 	 * @param numberIntegration The int value to which numberIntegration is to be set.
 	 
 		
 			
 	 */
 	void setNumberIntegration (int numberIntegration);
  		
	
	
	


	
	// ===> Attribute numberSubintegration
	
	
	

	
 	/**
 	 * Get numberSubintegration.
 	 * @return numberSubintegration as vector<int >
 	 */
 	vector<int > getNumberSubintegration() const;
	
 
 	
 	
 	/**
 	 * Set numberSubintegration with the specified vector<int >.
 	 * @param numberSubintegration The vector<int > value to which numberSubintegration is to be set.
 	 
 		
 			
 	 */
 	void setNumberSubintegration (vector<int > numberSubintegration);
  		
	
	
	


	
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
  		
	
	
	


	
	// ===> Attribute fieldName
	
	
	

	
 	/**
 	 * Get fieldName.
 	 * @return fieldName as string
 	 */
 	string getFieldName() const;
	
 
 	
 	
 	/**
 	 * Set fieldName with the specified string.
 	 * @param fieldName The string value to which fieldName is to be set.
 	 
 		
 			
 	 */
 	void setFieldName (string fieldName);
  		
	
	
	


	
	// ===> Attribute correlatorCalibration, which is optional
	
	
	
	/**
	 * The attribute correlatorCalibration is optional. Return true if this attribute exists.
	 * @return true if and only if the correlatorCalibration attribute exists. 
	 */
	bool isCorrelatorCalibrationExists() const;
	

	
 	/**
 	 * Get correlatorCalibration, which is optional.
 	 * @return correlatorCalibration as CorrelatorCalibrationMod::CorrelatorCalibration
 	 * @throws IllegalAccessException If correlatorCalibration does not exist.
 	 */
 	CorrelatorCalibrationMod::CorrelatorCalibration getCorrelatorCalibration() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set correlatorCalibration with the specified CorrelatorCalibrationMod::CorrelatorCalibration.
 	 * @param correlatorCalibration The CorrelatorCalibrationMod::CorrelatorCalibration value to which correlatorCalibration is to be set.
 	 
 		
 	 */
 	void setCorrelatorCalibration (CorrelatorCalibrationMod::CorrelatorCalibration correlatorCalibration);
		
	
	
	
	/**
	 * Mark correlatorCalibration, which is an optional field, as non-existent.
	 */
	void clearCorrelatorCalibration ();
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute execBlockId
	
	
	

	
 	/**
 	 * Get execBlockId.
 	 * @return execBlockId as Tag
 	 */
 	Tag getExecBlockId() const;
	
 
 	
 	
 	/**
 	 * Set execBlockId with the specified Tag.
 	 * @param execBlockId The Tag value to which execBlockId is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setExecBlockId (Tag execBlockId);
  		
	
	
	


	
	// ===> Attribute scanNumber
	
	
	

	
 	/**
 	 * Get scanNumber.
 	 * @return scanNumber as int
 	 */
 	int getScanNumber() const;
	
 
 	
 	
 	/**
 	 * Set scanNumber with the specified int.
 	 * @param scanNumber The int value to which scanNumber is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setScanNumber (int scanNumber);
  		
	
	
	


	///////////
	// Links //
	///////////
	
	

	
		
	/**
	 * execBlockId pointer to the row in the ExecBlock table having ExecBlock.execBlockId == execBlockId
	 * @return a ExecBlockRow*
	 * 
	 
	 */
	 ExecBlockRow* getExecBlockUsingExecBlockId();
	 

	

	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this SubscanRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(Tag execBlockId, int scanNumber, int subscanNumber, ArrayTime startTime, ArrayTime endTime, SubscanIntentMod::SubscanIntent subscanIntent, int numberIntegration, vector<int > numberSubintegration, bool flagRow, string fieldName);
	
	

	
	bool compareRequiredValue(ArrayTime startTime, ArrayTime endTime, SubscanIntentMod::SubscanIntent subscanIntent, int numberIntegration, vector<int > numberSubintegration, bool flagRow, string fieldName); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the SubscanRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(SubscanRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	SubscanTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a SubscanRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	SubscanRow (SubscanTable &table);

	/**
	 * Create a SubscanRow using a copy constructor mechanism.
	 * <p>
	 * Given a SubscanRow row and a SubscanTable table, the method creates a new
	 * SubscanRow owned by table. Each attribute of the created row is a copy (deep)
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
	 SubscanRow (SubscanTable &table, SubscanRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute subscanNumber
	
	

	int subscanNumber;

	
	
 	

	
	// ===> Attribute startTime
	
	

	ArrayTime startTime;

	
	
 	

	
	// ===> Attribute endTime
	
	

	ArrayTime endTime;

	
	
 	

	
	// ===> Attribute subscanIntent
	
	

	SubscanIntentMod::SubscanIntent subscanIntent;

	
	
 	

	
	// ===> Attribute subscanMode, which is optional
	
	
	bool subscanModeExists;
	

	SwitchingModeMod::SwitchingMode subscanMode;

	
	
 	

	
	// ===> Attribute numberIntegration
	
	

	int numberIntegration;

	
	
 	

	
	// ===> Attribute numberSubintegration
	
	

	vector<int > numberSubintegration;

	
	
 	

	
	// ===> Attribute flagRow
	
	

	bool flagRow;

	
	
 	

	
	// ===> Attribute fieldName
	
	

	string fieldName;

	
	
 	

	
	// ===> Attribute correlatorCalibration, which is optional
	
	
	bool correlatorCalibrationExists;
	

	CorrelatorCalibrationMod::CorrelatorCalibration correlatorCalibration;

	
	
 	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute execBlockId
	
	

	Tag execBlockId;

	
	
 	

	
	// ===> Attribute scanNumber
	
	

	int scanNumber;

	
	
 	

	///////////
	// Links //
	///////////
	
	
		

	 

	


};

} // End namespace asdm

#endif /* Subscan_CLASS */
