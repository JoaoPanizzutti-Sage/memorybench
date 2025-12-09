/*
LoCoMo Search Script
Searches for memories for a single conversation's QA pairs.

Usage: bun run search.ts <conversationId> <runId> [waitSeconds]
*/

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { config, validateConfig } from '../utils/config.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

validateConfig(['apiKey', 'baseUrl']);

// Get command line arguments
const args = process.argv.slice(2);
if (args.length < 2) {
    console.error("Usage: bun run search.ts <conversationId> <runId> [waitSeconds]");
    process.exit(1);
}

const conversationId = args[0]!;
const runId = args[1]!;
const waitSeconds = args.length >= 3 ? parseInt(args[2]!) : 0;
const containerTag = `${conversationId}-${runId}`;

console.log(`\n=== LoCoMo Search ===`);
console.log(`Conversation ID: ${conversationId}`);
console.log(`Container Tag: ${containerTag}`);
if (waitSeconds > 0) {
    console.log(`Will wait ${waitSeconds}s for indexing`);
}
console.log('\n');

// Setup paths
const resultsDir = join(__dirname, '../../results');
if (!existsSync(resultsDir)) {
    mkdirSync(resultsDir, { recursive: true });
}

const resultsFilePath = join(resultsDir, `search-${conversationId}-${runId}.json`);

// Load conversation data
const locomoDataPath = join(__dirname, '../../locomo10.json');
if (!existsSync(locomoDataPath)) {
    console.error(`Error: locomo10.json not found at ${locomoDataPath}`);
    process.exit(1);
}

const locomoData = JSON.parse(readFileSync(locomoDataPath, 'utf8'));
const conversation = locomoData.find((c: any) => c.sample_id === conversationId);

if (!conversation) {
    console.error(`Error: Conversation ${conversationId} not found`);
    process.exit(1);
}

const qaPairs = conversation.qa || [];
console.log(`Found ${qaPairs.length} QA pairs to search\n`);

// Search function
const searchQuestion = async (question: string): Promise<any> => {
    const response = await fetch(`${config.baseUrl}/v4/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify({
            q: question,
            containerTag: containerTag,
            limit: 10,
            threshold: 0.3,
            include: { chunks: true }
        }),
    });

    if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
    }

    return response.json();
};

// Main search function
const runSearch = async () => {
    if (waitSeconds > 0) {
        console.log(`Waiting ${waitSeconds} seconds for indexing...`);
        await new Promise(resolve => setTimeout(resolve, waitSeconds * 1000));
        console.log('Proceeding with search...\n');
    }

    const results: any[] = [];

    for (let i = 0; i < qaPairs.length; i++) {
        const qa = qaPairs[i];
        const { question, answer, evidence, category } = qa;

        console.log(`[${i + 1}/${qaPairs.length}] Searching...`);
        console.log(`  Q: ${question.substring(0, 60)}${question.length > 60 ? '...' : ''}`);

        try {
            const searchResults = await searchQuestion(question);
            const resultCount = searchResults.results?.length || 0;
            console.log(`  Found ${resultCount} results`);

            results.push({
                conversation_id: conversationId,
                question,
                question_category: category,
                question_date: qa.question_date,
                ground_truth_answer: answer,
                evidence,
                search_results: searchResults,
                timestamp: new Date().toISOString(),
            });

            // Save intermediate results
            writeFileSync(resultsFilePath, JSON.stringify({
                metadata: {
                    conversation_id: conversationId,
                    container_tag: containerTag,
                    run_id: runId,
                    questions_processed: results.length,
                    total_questions: qaPairs.length,
                    status: 'in_progress',
                    timestamp: new Date().toISOString(),
                },
                results
            }, null, 2));

        } catch (error) {
            console.error(`  Error: ${error}`);
            results.push({
                conversation_id: conversationId,
                question,
                question_category: category,
                ground_truth_answer: answer,
                evidence,
                search_results: null,
                error: error instanceof Error ? error.message : String(error),
                timestamp: new Date().toISOString(),
            });
        }

        await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Save final results
    writeFileSync(resultsFilePath, JSON.stringify({
        metadata: {
            conversation_id: conversationId,
            container_tag: containerTag,
            run_id: runId,
            total_questions: qaPairs.length,
            status: 'completed',
            timestamp: new Date().toISOString(),
        },
        results
    }, null, 2));

    console.log(`\n✓ Search results saved to: ${resultsFilePath}`);
};

await runSearch();

