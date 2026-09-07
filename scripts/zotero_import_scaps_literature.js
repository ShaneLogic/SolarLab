// Run in Zotero via Tools > Developer > Run JavaScript.
// Imports verified photovoltaic references into the SCAPS collection.

const parentCollectionKey = "NDVBQ6YJ";
const parentCollectionName = "SCAPS";
const libraryID = Zotero.Libraries.userLibraryID;

const works = [
  {
    label: "2013 vapour-deposited planar heterojunction",
    identifier: { DOI: "10.1038/nature12509" },
    expectedTitle:
      "Efficient planar heterojunction perovskite solar cells by vapour deposition",
    titleNeedle:
      "efficient planar heterojunction perovskite solar cells by vapour deposition",
    pdfURLs: [
      "https://www.epfl.ch/labs/limno/wp-content/uploads/2018/08/5.pdf",
      "https://www.nature.com/articles/nature12509.pdf",
    ],
    pdfBaseName: "Liu_2013_vapour_deposited_planar_heterojunction",
  },
  {
    label: "QF transport",
    identifier: { DOI: "10.1016/j.electacta.2021.138696" },
    expectedTitle:
      "Modelling charge transport in perovskite solar cells: Potential-based and limiting ion depletion",
    titleNeedle: "modelling charge transport in perovskite solar cells",
    pdfURLs: [
      "https://oa.tib.eu/renate/bitstreams/9cbcdb2a-5710-4101-82e2-f2515f7ccbc0/download",
    ],
    pdfBaseName: "Abdel_2021_QF_transport",
  },
  {
    label: "Bayesian ion inference",
    identifier: { DOI: "10.1088/2515-7655/ad0a38" },
    expectedTitle:
      "Bayesian parameter estimation for characterising mobile ion vacancies in perovskite solar cells",
    titleNeedle:
      "bayesian parameter estimation for characterising mobile ion vacancies",
    pdfURLs: ["https://arxiv.org/pdf/2309.14302"],
    pdfBaseName: "McCallum_2024_Bayesian_ion_inference",
  },
  {
    label: "2024 impedance",
    identifier: { DOI: "10.1002/aenm.202400955" },
    expectedTitle:
      "Understanding the Full Zoo of Perovskite Solar Cell Impedance Spectra with the Standard Drift-Diffusion Model",
    titleNeedle:
      "understanding the full zoo of perovskite solar cell impedance spectra",
    pdfURLs: [
      "https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/65bcd96666c1381729ce74e9/original/understanding-the-full-zoo-of-perovskite-solar-cell-impedance-spectra-with-the-standard-drift-diffusion-model.pdf",
      "https://eprints.soton.ac.uk/492174/2/Advanced_Energy_Materials_-_2024_-_Clarke_-_Understanding_the_Full_Zoo_of_Perovskite_Solar_Cell_Impedance_Spectra_with_the.pdf",
      "https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/aenm.202400955",
    ],
    pdfBaseName: "Clarke_2024_Full_Zoo_impedance",
  },
  {
    label: "2025 generalized SG",
    identifier: { DOI: "10.1007/s10825-025-02412-4" },
    expectedTitle:
      "Thermodynamically consistent stabilization of the drift-diffusion model for arbitrary band structures and carrier statistics",
    titleNeedle:
      "thermodynamically consistent stabilization of the drift-diffusion model",
    pdfURLs: [
      "https://link.springer.com/content/pdf/10.1007/s10825-025-02412-4.pdf",
      "https://publications.rwth-aachen.de/record/1019029/files/1019029.pdf",
    ],
    pdfBaseName: "Linn_2025_generalized_SG",
  },
  {
    label: "2025 ion characterization",
    identifier: { DOI: "10.1103/mr3l-jg9h" },
    expectedTitle:
      "Characterization of Mobile Ions in Perovskite Solar Cells with Capacitance and Current Measurements by Approximating Drift-Diffusion Simulations",
    titleNeedle: "characterization of mobile ions in perovskite solar cells",
    pdfURLs: ["https://ir.amolf.nl/pub/11252/17147publishedVersion.pdf"],
    pdfBaseName: "Schmidt_2025_mobile_ion_characterization",
  },
  {
    label: "2026 Radau-SG",
    identifier: { arXiv: "2603.09063" },
    expectedTitle:
      "A Stable, High-Order Time-Stepping Scheme for the Drift-Diffusion Model in Modern Solar Cell Simulation",
    titleNeedle:
      "stable, high-order time-stepping scheme for the drift-diffusion model",
    pdfURLs: ["https://arxiv.org/pdf/2603.09063"],
    pdfBaseName: "Du_Yan_2026_Radau_SG",
  },
];

const normalize = (value) =>
  Zotero.Utilities.cleanTags(String(value || ""))
    .replace(/[\u2010-\u2015]/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();

const parent = await Zotero.Collections.getByLibraryAndKeyAsync(
  libraryID,
  parentCollectionKey,
);
if (!parent || parent.name !== parentCollectionName) {
  throw new Error(
    `Expected ${parentCollectionName} collection with key ${parentCollectionKey}`,
  );
}

async function searchItemIDs(condition, operator, value) {
  const search = new Zotero.Search();
  search.libraryID = libraryID;
  search.addCondition(condition, operator, value);
  return await search.search();
}

async function findExisting(work) {
  const ids = new Set();

  if (work.identifier.DOI) {
    for (const id of await searchItemIDs("DOI", "is", work.identifier.DOI)) {
      ids.add(id);
    }
  }
  if (work.identifier.arXiv) {
    for (const [condition, operator] of [
      ["archiveID", "contains"],
      ["url", "contains"],
    ]) {
      for (const id of await searchItemIDs(
        condition,
        operator,
        work.identifier.arXiv,
      )) {
        ids.add(id);
      }
    }
  }
  for (const id of await searchItemIDs("title", "is", work.expectedTitle)) {
    ids.add(id);
  }

  const items = [];
  for (const id of ids) {
    const item = await Zotero.Items.getAsync(id);
    if (item && !item.deleted && item.isTopLevelItem()) items.push(item);
  }
  if (items.length > 1) {
    throw new Error(
      `${work.label} matches multiple existing Zotero items: ${items
        .map((item) => item.key)
        .join(", ")}`,
    );
  }
  return items[0] || null;
}

function validateMetadata(work, item) {
  const title = item.getField("title");
  if (!normalize(title).includes(normalize(work.titleNeedle))) {
    throw new Error(
      `${work.label} title mismatch: expected ${work.expectedTitle}; got ${title}`,
    );
  }

  if (work.identifier.DOI) {
    const actualDOI = Zotero.Utilities.cleanDOI(item.getField("DOI") || "");
    if (normalize(actualDOI) !== normalize(work.identifier.DOI)) {
      throw new Error(
        `${work.label} DOI mismatch: expected ${work.identifier.DOI}; got ${actualDOI}`,
      );
    }
  }
}

function hasPDFAttachment(item) {
  for (const attachmentID of item.getAttachments(false)) {
    const attachment = Zotero.Items.get(attachmentID);
    if (attachment && !attachment.deleted && attachment.isPDFAttachment()) {
      return true;
    }
  }
  return false;
}

async function importMetadata(work) {
  const translate = new Zotero.Translate.Search();
  translate.setIdentifier(work.identifier);
  const translators = await translate.getTranslators();
  if (!translators.length) {
    throw new Error(`No Zotero lookup translator found for ${work.label}`);
  }
  translate.setTranslator(translators);

  const translatedItems = await translate.translate({
    libraryID,
    collections: [parent.id],
    saveAttachments: false,
  });
  const topLevelItems = translatedItems.filter(
    (item) => item && !item.deleted && item.isTopLevelItem(),
  );
  if (topLevelItems.length !== 1) {
    throw new Error(
      `${work.label} lookup returned ${topLevelItems.length} top-level items`,
    );
  }
  return topLevelItems[0];
}

// Fail before writes if the live library already contains ambiguous duplicates.
const preflight = new Map();
for (const work of works) {
  preflight.set(work.label, await findExisting(work));
}

const result = {
  parent: `${parent.name} (${parent.key})`,
  created: [],
  reused: [],
  addedToSCAPS: [],
  pdfAttached: [],
  pdfAlreadyPresent: [],
  pdfErrors: [],
  verification: [],
};

for (const work of works) {
  let item = preflight.get(work.label);
  if (item) {
    result.reused.push(work.label);
  } else {
    item = await importMetadata(work);
    result.created.push(work.label);
  }

  validateMetadata(work, item);

  if (!item.inCollection(parent.id)) {
    item.addToCollection(parent.id);
    await item.saveTx({ skipDateModifiedUpdate: true });
    result.addedToSCAPS.push(work.label);
  }

  if (hasPDFAttachment(item)) {
    result.pdfAlreadyPresent.push(work.label);
    continue;
  }

  const attemptErrors = [];
  let pdfAdded = false;
  for (const url of work.pdfURLs) {
    try {
      await Zotero.Attachments.importFromURL({
        libraryID,
        url,
        parentItemID: item.id,
        title: "Open-access full text PDF",
        fileBaseName: work.pdfBaseName,
        contentType: "application/pdf",
      });
      pdfAdded = true;
      break;
    } catch (error) {
      Zotero.logError(error);
      attemptErrors.push({ url, error: error.message || String(error) });
    }
  }

  if (!pdfAdded) {
    try {
      const attachment = await Zotero.Attachments.addAvailableFile(item);
      pdfAdded = Boolean(attachment && attachment.isPDFAttachment());
      if (!pdfAdded) {
        attemptErrors.push({
          url: "Zotero.Attachments.addAvailableFile",
          error: "No PDF was found",
        });
      }
    } catch (error) {
      Zotero.logError(error);
      attemptErrors.push({
        url: "Zotero.Attachments.addAvailableFile",
        error: error.message || String(error),
      });
    }
  }

  if (pdfAdded) {
    result.pdfAttached.push(work.label);
  } else {
    result.pdfErrors.push({ label: work.label, attempts: attemptErrors });
  }
}

for (const work of works) {
  const item = await findExisting(work);
  if (!item) throw new Error(`${work.label} is missing after import`);
  validateMetadata(work, item);
  if (!item.inCollection(parent.id)) {
    throw new Error(`${work.label} is not in the SCAPS collection after import`);
  }
  result.verification.push({
    label: work.label,
    key: item.key,
    title: item.getField("title"),
    DOI: item.getField("DOI") || null,
    hasPDF: hasPDFAttachment(item),
  });
}

result.scapsDirectItems = parent.getChildItems(false, false).length;
result.complete =
  result.verification.length === works.length &&
  result.verification.every((entry) => entry.hasPDF) &&
  result.pdfErrors.length === 0;

return JSON.stringify(result, null, 2);
