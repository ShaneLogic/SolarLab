// Run in Zotero via Tools > Developer > Run JavaScript.
// Idempotently creates topical subcollections under the SCAPS collection.

const parentCollectionKey = "NDVBQ6YJ";
const parentCollectionName = "SCAPS";
const libraryID = Zotero.Libraries.userLibraryID;

// Item keys are ordered by first-author surname, then by year for one author.
const groups = {
  "Fundamentals/SCAPS": [
    "QRDDFI23", // Alshaikh 2026, PerovskiteOpt-AI
    "9359QD38", // Burgelman and Marlein 2008, graded band gap
    "CG3DZ7AD", // Pauwels and Vanhoutte 1978, interface foundations
    "IDCC325T", // Verschraegen and Burgelman 2007, SCAPS tunneling
  ],
  "Numerics": [
    "T3TC2BCR", // Calado et al. 2022, Driftfusion
    "2UMJ74BP", // Clarke et al. 2023, IonMonger 2.0
    "XCE83EKB", // Courtier et al. 2018, robust numerical scheme
    "7MWX3GXF", // Courtier et al. 2019, IonMonger
    "66ZG4W9Y", // Gao et al. 2025, PyPWDFT
    "RKSYPGWI", // Mann et al. 2022, differentiable PV simulator
    "DRKWULJJ", // Sachsenweger et al. 2026, ChargeFabrica
    "DC27D8WH", // Chebfun guide, top-level tool attachment
  ],
  "Interface & tunneling": [
    "NE5V97RJ", // Calado et al. 2016, ions and contact recombination
    "2YFBDD4R", // Courtier et al. 2019, transport-layer properties
    "KXBRNKIE", // Kogo et al. 2018, blocking layers
    "CG3DZ7AD", // Pauwels and Vanhoutte 1978, interface states/barriers
    "IDCC325T", // Verschraegen and Burgelman 2007, WKB tunneling
  ],
  "Mobile ions": [
    "NE5V97RJ", // Calado et al. 2016, experimental ion migration
    "T3TC2BCR", // Calado et al. 2022, Driftfusion
    "2UMJ74BP", // Clarke et al. 2023, IonMonger 2.0
    "XCE83EKB", // Courtier et al. 2018, ion-vacancy numerics
    "2YFBDD4R", // Courtier et al. 2019, transport layers and ions
    "7MWX3GXF", // Courtier et al. 2019, IonMonger
    "DRKWULJJ", // Sachsenweger et al. 2026, 2D electro-ionic DD
  ],
  "Impedance & inverse": [
    "2UMJ74BP", // Clarke et al. 2023, IonMonger 2.0 impedance
    "RKSYPGWI", // Mann et al. 2022, differentiable parameter discovery
    "PBMAWR7T", // Pitarch-Tena et al. 2018, impedance protocols
  ],
  "Multidimensional": [
    "DRKWULJJ", // Sachsenweger et al. 2026, ChargeFabrica
  ],
  "Differentiable/AI": [
    "QRDDFI23", // Alshaikh 2026, SCAPS surrogate/BO
    "RKSYPGWI", // Mann et al. 2022, implicit differentiation
  ],
  "Experimental validation": [
    "NE5V97RJ", // Calado et al. 2016, ion-migration evidence
    "KXBRNKIE", // Kogo et al. 2018, blocking-layer devices
    "5S4FJZSK", // Liu et al. 2013, vapor-deposited planar devices
    "PBMAWR7T", // Pitarch-Tena et al. 2018, impedance measurements
    "DRKWULJJ", // Sachsenweger et al. 2026, mesoporous-device comparison
  ],
};

const parent = await Zotero.Collections.getByLibraryAndKeyAsync(
  libraryID,
  parentCollectionKey,
);
if (!parent || parent.name !== parentCollectionName) {
  throw new Error(
    `Expected ${parentCollectionName} collection with key ${parentCollectionKey}`,
  );
}

const uniqueItemKeys = [...new Set(Object.values(groups).flat())];
const itemsByKey = new Map();
for (const key of uniqueItemKeys) {
  const item = await Zotero.Items.getByLibraryAndKeyAsync(libraryID, key);
  if (!item || item.deleted || !item.isTopLevelItem()) {
    throw new Error(`Missing, deleted, or non-top-level SCAPS item: ${key}`);
  }
  itemsByKey.set(key, item);
}

const parentItemIDs = new Set(parent.getChildItems(true, false));
const outsideParent = [...itemsByKey.entries()]
  .filter(([, item]) => !parentItemIDs.has(item.id))
  .map(([key]) => key);
if (outsideParent.length) {
  throw new Error(`Items are no longer direct SCAPS members: ${outsideParent.join(", ")}`);
}

const childrenByName = new Map();
for (const child of Zotero.Collections.getByParent(parent.id, false, false)) {
  if (childrenByName.has(child.name)) {
    throw new Error(`Duplicate SCAPS subcollection name: ${child.name}`);
  }
  childrenByName.set(child.name, child);
}

const created = [];
let membershipsAdded = 0;
await Zotero.DB.executeTransaction(async () => {
  for (const name of Object.keys(groups)) {
    if (childrenByName.has(name)) continue;

    const collection = new Zotero.Collection();
    collection.libraryID = libraryID;
    collection.name = name;
    collection.parentID = parent.id;
    await collection.save();
    childrenByName.set(name, collection);
    created.push(name);
  }

  for (const [name, itemKeys] of Object.entries(groups)) {
    const collection = childrenByName.get(name);
    for (const key of itemKeys) {
      const item = itemsByKey.get(key);
      if (item.inCollection(collection.id)) continue;
      item.addToCollection(collection.id);
      await item.save({ skipDateModifiedUpdate: true });
      membershipsAdded++;
    }
  }
});

const verification = {};
for (const [name, expectedKeys] of Object.entries(groups)) {
  const collection = childrenByName.get(name);
  const actualItems = collection.getChildItems(false, false);
  const actualKeys = new Set(actualItems.map((item) => item.key));
  const missing = expectedKeys.filter((key) => !actualKeys.has(key));
  if (missing.length) {
    throw new Error(`${name} verification failed; missing: ${missing.join(", ")}`);
  }
  verification[name] = {
    expected: expectedKeys.length,
    actual: actualItems.length,
    titles: actualItems
      .map((item) => item.getField("title") || item.attachmentFilename || item.key)
      .sort((a, b) => a.localeCompare(b)),
  };
}

return JSON.stringify(
  {
    parent: `${parent.name} (${parent.key})`,
    created,
    membershipsAdded,
    verification,
  },
  null,
  2,
);
