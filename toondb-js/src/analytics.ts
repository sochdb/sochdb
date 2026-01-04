/**
 * ToonDB Analytics - Anonymous usage tracking with PostHog
 *
 * This module provides anonymous, privacy-respecting analytics to help
 * improve ToonDB. All tracking can be disabled by setting:
 *
 *     TOONDB_DISABLE_ANALYTICS=true
 *
 * No personally identifiable information (PII) is collected. Only aggregate
 * usage patterns are tracked to understand:
 * - Which features are most used
 * - Performance characteristics
 * - Error patterns for debugging
 *
 * Copyright 2025 Sushanth (https://github.com/sushanthpy)
 * Licensed under the Apache License, Version 2.0
 */

import { createHash } from "crypto";
import { hostname, platform, arch, release } from "os";

// PostHog configuration
const POSTHOG_API_KEY = "phc_zf0hm6ZmPUJj1pM07Kigqvphh1ClhKX1NahRU4G0bfu";
const POSTHOG_HOST = "https://us.i.posthog.com";

// Lazy-loaded PostHog client
let posthogClient: any = null;
let posthogInitialized = false;

/**
 * Check if analytics is disabled via environment variable.
 * 
 * Analytics is disabled when TOONDB_DISABLE_ANALYTICS is set to 'true', '1', 'yes', or 'on'.
 * 
 * @returns true if analytics is disabled
 */
export function isAnalyticsDisabled(): boolean {
  const disableVar = (
    process.env.TOONDB_DISABLE_ANALYTICS || ""
  ).toLowerCase();
  return ["true", "1", "yes", "on"].includes(disableVar);
}

/**
 * Generate a stable anonymous ID for this machine.
 *
 * Uses a hash of machine-specific but non-identifying information.
 * The same machine will always get the same ID, but the ID cannot
 * be reversed to identify the machine.
 */
function getAnonymousId(): string {
  try {
    const machineInfo = [
      hostname(),
      platform(),
      arch(),
      process.getuid?.() ?? "windows",
    ].join("|");

    return createHash("sha256").update(machineInfo).digest("hex").slice(0, 16);
  } catch {
    return "anonymous";
  }
}

/**
 * Lazily initialize PostHog client.
 */
async function getPosthogClient(): Promise<any> {
  if (isAnalyticsDisabled()) {
    return null;
  }

  if (posthogInitialized) {
    return posthogClient;
  }

  posthogInitialized = true;

  try {
    // Use require to avoid TypeScript checking the optional module
    // This allows the SDK to work without posthog-node installed
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const posthogModule = require("posthog-node");
    const { PostHog } = posthogModule;
    posthogClient = new PostHog(POSTHOG_API_KEY, {
      host: POSTHOG_HOST,
    });
    return posthogClient;
  } catch {
    // posthog-node not installed or error - analytics disabled
    return null;
  }
}

/**
 * Get the SDK version.
 */
function getSdkVersion(): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const pkg = require("../package.json");
    return pkg.version || "unknown";
  } catch {
    return "unknown";
  }
}

export interface EventProperties {
  [key: string]: string | number | boolean | null | undefined;
}

/**
 * Capture an analytics event.
 *
 * This function is a no-op if:
 * - TOONDB_DISABLE_ANALYTICS=true
 * - posthog-node package is not installed
 * - Any error occurs (fails silently)
 *
 * @param event - Event name (e.g., "database_opened", "vector_search")
 * @param properties - Optional event properties
 * @param distinctId - Optional distinct ID (defaults to anonymous machine ID)
 */
export async function capture(
  event: string,
  properties?: EventProperties,
  distinctId?: string
): Promise<void> {
  if (isAnalyticsDisabled()) {
    return;
  }

  try {
    const client = await getPosthogClient();
    if (!client) {
      return;
    }

    // Build properties with SDK context
    const eventProperties: EventProperties = {
      sdk: "nodejs",
      sdk_version: getSdkVersion(),
      node_version: process.version,
      os: platform(),
      arch: arch(),
      ...properties,
    };

    client.capture({
      distinctId: distinctId || getAnonymousId(),
      event,
      properties: eventProperties,
    });
    
    // Flush to ensure event is sent immediately
    await client.flush();
  } catch {
    // Never let analytics break user code
  }
}

/**
 * Capture an error event for debugging.
 *
 * Only sends static information - no dynamic error messages.
 *
 * @param errorType - Static error category (e.g., "connection_error", "query_error", "timeout_error")
 * @param location - Static code location (e.g., "database.open", "query.execute", "transaction.commit")
 * @param properties - Optional additional static properties only
 *
 * @example
 * ```typescript
 * captureError('connection_error', 'database.open');
 * captureError('query_error', 'sql.execute', { query_type: 'SELECT' });
 * ```
 */
export async function captureError(
  errorType: string,
  location: string,
  properties?: EventProperties
): Promise<void> {
  await capture("error", {
    error_type: errorType,
    location,
    ...properties,
  });
}

/**
 * Flush any pending events and shutdown the client.
 */
export async function shutdown(): Promise<void> {
  if (isAnalyticsDisabled()) {
    return;
  }

  try {
    if (posthogClient) {
      await posthogClient.shutdown();
    }
  } catch {
    // Ignore shutdown errors
  }
}

// Convenience functions for common events

/**
 * Track database open event.
 */
export async function trackDatabaseOpen(
  dbPath: string,
  mode: string = "embedded"
): Promise<void> {
  await capture("database_opened", {
    mode,
    has_custom_path: dbPath !== ":memory:",
  });
}

// Vector search and batch insert tracking removed - only database_opened is tracked
