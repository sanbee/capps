
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
 * File Representable.h
 */
 
#ifndef _Representable_H
#define _Representable_H

#ifndef __cplusplus
#error This is a C++ include file and cannot be used from plain C
#endif
 
#include <Entity.h>
#include <ConversionException.h>

using std::string;
using asdm::Entity;
using asdm::ConversionException;

namespace asdm {


/**
 * The Representable interface is implemented by all tables
 * and by the container.  It represents methods that are 
 * required to convert between the internal and the various 
 * external representations of tables and containers.
 */
class Representable {

public:

	virtual ~Representable() {}
	virtual string toXML()   = 0;
	virtual void fromXML(string xml) = 0;

	virtual string toVOTable() const throw(ConversionException) = 0;
	virtual void fromVOTable(string vo) throw(ConversionException) = 0;

	virtual char *toFITS() const throw(ConversionException) = 0;
	virtual void fromFITS(char *fits) throw(ConversionException) = 0;

	virtual void setEntity(Entity e) = 0;
	virtual Entity getEntity() const = 0;

	virtual string getName() const = 0;
	virtual unsigned int size() = 0;

};
	
} // End namespace asdm

#endif /* _Representable_CLASS */
