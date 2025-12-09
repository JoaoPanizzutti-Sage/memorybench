/*
LoCoMo Ingestion Script
Ingests a single conversation session-by-session with checkpoint support.

Usage: bun run ingest.ts <conversationId> <runId>
Example: bun run ingest.ts conv-26 eval1
*/

import { readFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { config, validateConfig } from '../utils/config.ts';
import { CheckpointManager, type CheckpointData } from '../utils/checkpoint.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

validateConfig(['apiKey', 'baseUrl']);

// Get command line arguments
const args = process.argv.slice(2);
if (args.length < 2) {
    console.error("Usage: bun run ingest.ts <conversationId> <runId>");
    console.error("Example: bun run ingest.ts conv-26 eval1");
    process.exit(1);
}

const conversationId = args[0]!;
const runId = args[1]!;
const containerTag = `${conversationId}-${runId}`;

console.log(`\n=== LoCoMo Ingestion ===`);
console.log(`Conversation ID: ${conversationId}`);
console.log(`Container Tag: ${containerTag}\n`);

// Setup paths
const sessionsDir = join(__dirname, '../../data/sessions');
const checkpointManager = new CheckpointManager(join(__dirname, '../../checkpoints'));

if (!existsSync(sessionsDir)) {
    console.error(`Error: Sessions directory not found at ${sessionsDir}`);
    process.exit(1);
}

// Find all session files for this conversation
const sessionFiles = readdirSync(sessionsDir)
    .filter((f: string) => f.startsWith(`${conversationId}-session-`) && f.endsWith('.json'))
    .sort((a: string, b: string) => {
        const aNum = parseInt(a.match(/session-(\d+)/)?.[1] || '0');
        const bNum = parseInt(b.match(/session-(\d+)/)?.[1] || '0');
        return aNum - bNum;
    });

if (sessionFiles.length === 0) {
    console.error(`Error: No session files found for ${conversationId} in ${sessionsDir}`);
    process.exit(1);
}

console.log(`Found ${sessionFiles.length} session files for ${conversationId}`);

// Load or initialize checkpoint
let checkpoint: CheckpointData = checkpointManager.loadCheckpoint(conversationId, runId) || {
    conversationId,
    runId,
    containerTag,
    totalSessions: sessionFiles.length,
    sessions: []
};

if (checkpoint.sessions.length === 0) {
    console.log(`Creating new checkpoint\n`);
    
    // Initialize all sessions as pending
    for (const sessionFile of sessionFiles) {
        const sessionPath = join(sessionsDir, sessionFile);
        const sessionData = JSON.parse(readFileSync(sessionPath, 'utf8'));
        const turns = sessionData.conversation_turns || [];

        checkpoint.sessions.push({
            sessionNumber: sessionData.session_number,
            sessionKey: sessionData.session_id,
            turnsTotal: turns.length,
            turnsIngested: 0,
            status: 'pending'
        });
    }
} else {
    console.log(`Loading existing checkpoint\n`);
}

// Ingest a single session as one document
const ingestSession = async (
    turns: any[],
    sessionKey: string,
    sessionDate: string
): Promise<boolean> => {
    const allTurnsText = turns.map(turn => {
        const speaker = turn.speaker || 'Unknown';
        const diaId = turn.dia_id || '';
        const text = turn.text || '';
        return `[${diaId}] ${speaker}: ${text}`;
    }).join('\n');

    const content = `Session: ${sessionKey}\nDate: ${sessionDate}\n\n${allTurnsText}`;

    try {
        const response = await fetch(`${config.baseUrl}/v3/documents`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`,
            },
            body: JSON.stringify({
                content,
                containerTag: containerTag,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        return true;
    } catch (error) {
        console.error(`  Error: ${error instanceof Error ? error.message : String(error)}`);
        return false;
    }
};

// Main ingestion function
const runIngestion = async () => {
    console.log(`Found ${checkpoint.totalSessions} sessions to ingest\n`);
    console.log('─'.repeat(60));

    let completedCount = 0;
    let failedCount = 0;
    let skippedCount = 0;

    for (let i = 0; i < checkpoint.sessions.length; i++) {
        const session = checkpoint.sessions[i]!;
        
        if (session.status === 'completed') {
            console.log(`Session ${i + 1}: Already completed, skipping`);
            skippedCount++;
            continue;
        }

        console.log(`\nSession ${i + 1}/${checkpoint.sessions.length}: ${session.sessionKey}`);
        session.status = 'in_progress';

        // Load session data
        const sessionFileName = `${session.sessionKey}.json`;
        const sessionPath = join(sessionsDir, sessionFileName);

        if (!existsSync(sessionPath)) {
            console.error(`  Session file not found: ${sessionPath}`);
            session.status = 'failed';
            session.error = 'Session file not found';
            checkpointManager.saveCheckpoint(checkpoint);
            failedCount++;
            continue;
        }

        const sessionData = JSON.parse(readFileSync(sessionPath, 'utf8'));
        const turns = sessionData.conversation_turns || [];
        const sessionDate = sessionData.events?.date || '';

        const success = await ingestSession(turns, session.sessionKey, sessionDate);

        if (success) {
            console.log(`  ✓ Successfully ingested (${turns.length} turns)`);
            session.status = 'completed';
            session.turnsIngested = turns.length;
            session.timestamp = new Date().toISOString();
            completedCount++;
        } else {
            session.status = 'failed';
            session.error = 'Failed to ingest';
            failedCount++;
        }

        checkpointManager.saveCheckpoint(checkpoint);

        // Wait between sessions
        if (i < checkpoint.sessions.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }

    console.log('\n' + '─'.repeat(60));
    console.log('\n=== Ingestion Summary ===');
    console.log(`Completed: ${completedCount + skippedCount}`);
    console.log(`Failed: ${failedCount}`);
    console.log(`Skipped: ${skippedCount}`);

    return failedCount === 0;
};

// Run
const success = await runIngestion();
process.exit(success ? 0 : 1);

